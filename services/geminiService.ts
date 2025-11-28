import { GoogleGenAI, Chat, GenerateContentResponse, HarmCategory, HarmBlockThreshold } from "@google/genai";
import { EVE_SYSTEM_INSTRUCTION, MODELS, EVE_APPEARANCE, EVE_REFERENCE_IMAGES } from '../constants';
import { ModelTier, Message } from '../types';

// Singleton chat instance to maintain history during the session
let chatSession: Chat | null = null;
let currentTier: ModelTier = 'free';

// Define the most permissive safety settings allowed by the API
// BLOCK_NONE is critical for artistic anatomy and high-fashion photography to prevent false positives on skin exposure.
const SAFETY_SETTINGS = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: 'HARM_CATEGORY_CIVIC_INTEGRITY' as HarmCategory, threshold: HarmBlockThreshold.BLOCK_NONE },
];

export const initializeChat = (tier: ModelTier = 'free', history?: any[], apiKey?: string) => {
  try {
    const key = apiKey || process.env.API_KEY || '';
    if (!key) {
        console.warn("[System] No API Key provided. Chat may fail.");
    }
    const ai = new GoogleGenAI({ apiKey: key });
    currentTier = tier;
    
    // Validate history to prevent SDK crashes
    const validHistory = Array.isArray(history) ? history : [];

    chatSession = ai.chats.create({
      model: MODELS[tier].chat,
      config: {
        systemInstruction: EVE_SYSTEM_INSTRUCTION,
        temperature: 1.0,
        topP: 0.95,
        topK: 40,
        safetySettings: SAFETY_SETTINGS,
        // Speed Optimization: Disable thinking for free tier (Gemini 2.5 Flash) to reduce latency
        ...(tier === 'free' ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
      },
      history: validHistory,
    });
    console.log(`[System] Eve v2.0 Initialized on ${tier} tier with ${validHistory.length} context items.`);
  } catch (error) {
    console.error("Failed to initialize chat session with history:", error);
    // Fallback to empty session if history is corrupt
    const key = apiKey || process.env.API_KEY || '';
    const ai = new GoogleGenAI({ apiKey: key });
    chatSession = ai.chats.create({
      model: MODELS[tier].chat,
      config: { 
        systemInstruction: EVE_SYSTEM_INSTRUCTION,
        safetySettings: SAFETY_SETTINGS,
        ...(tier === 'free' ? { thinkingConfig: { thinkingBudget: 0 } } : {}),
      },
    });
  }
};

// Helper to determine intent
const isImageGenerationIntent = (text: string): boolean => {
  const keywords = ['generate', 'create', 'draw', 'imagine', 'render', 'visualize', 'make an image', 'image of', 'picture of'];
  const lower = text.toLowerCase();
  return keywords.some(k => lower.includes(k));
};

const isImageEditingIntent = (text: string): boolean => {
  const keywords = ['edit', 'change', 'filter', 'style', 'make it', 'turn it', 'add', 'remove', 'background', 'modify', 'bananafy'];
  const lower = text.toLowerCase();
  return keywords.some(k => lower.includes(k));
};

// Helper to optimize Cloudinary URLs to prevent payload too large errors (Error Code 6)
const optimizeCloudinaryUrl = (url: string): string => {
  if (url.includes('res.cloudinary.com') && url.includes('/upload/')) {
    // Inject resize and quality parameters
    // w_1024: Resize width to 1024px (keeps aspect ratio)
    // q_auto: Automatic quality compression
    // f_auto: Automatic format (usually WebP/AVIF which is smaller)
    // This reduces a 5MB PNG to ~100KB WebP
    return url.replace('/upload/', '/upload/w_1024,q_auto,f_auto/');
  }
  return url;
};

/**
 * Separate async function for generating selfies based on triggers.
 * This prevents the main chat loop from blocking while waiting for images.
 */
export const generateVisualSelfie = async (
    description: string, 
    tier: ModelTier, 
    apiKey?: string
): Promise<string | undefined> => {
    const key = apiKey || process.env.API_KEY || '';
    const ai = new GoogleGenAI({ apiKey: key });
    const imageModel = MODELS[tier].image;

    try {
        // COMPOSITE PROMPT: Explicitly reference the attached image for likeness
        // Framing as a "fictional character" and "artistic study" helps reduce false positives in safety filters.
        const selfiePrompt = `Generate a high-quality photorealistic portrait of this fictional character.
        
        CRITICAL: Preserve facial features, skin tone, hair style from the reference image.
        
        Scene/Action: ${description}.
        Character Details: ${EVE_APPEARANCE}
        
        Style: 8k resolution, cinematic lighting, raw photo, highly detailed, 9:16 aspect ratio, artistic fashion photography.`;
        
        console.log(`[System] Generating Selfie with prompt: ${description}`);

        let parts: any[] = [{ text: selfiePrompt }];

        // Try to fetch reference images to guide generation (Image-to-Image style)
        if (EVE_REFERENCE_IMAGES && EVE_REFERENCE_IMAGES.length > 0) {
            // Only select ONE random reference image to avoid XHR/Payload size errors
            const randomIndex = Math.floor(Math.random() * EVE_REFERENCE_IMAGES.length);
            const originalUrl = EVE_REFERENCE_IMAGES[randomIndex];
            // OPTIMIZE: Resize image URL to prevent "Error Code 6" (Payload Too Large)
            const optimizedUrl = optimizeCloudinaryUrl(originalUrl);

            console.log(`[System] Attaching reference image: ${optimizedUrl}`);
            
            try {
                const imgRes = await fetch(optimizedUrl);
                if (imgRes.ok) {
                    const blob = await imgRes.blob();
                    const arrayBuffer = await blob.arrayBuffer();
                    
                    // Prepend the image part
                    parts.unshift({
                        inlineData: {
                            mimeType: blob.type || 'image/png',
                            data: arrayBufferToBase64(arrayBuffer)
                        }
                    });
                } else {
                    console.warn(`[System] Failed to fetch reference image: ${imgRes.statusText}`);
                }
            } catch (e) {
                console.warn(`[System] Failed to load reference image: ${optimizedUrl}`, e);
            }
        }

        const selfieResponse = await ai.models.generateContent({
          model: imageModel,
          contents: { parts: parts },
          config: {
            imageConfig: { aspectRatio: '9:16' },
            safetySettings: SAFETY_SETTINGS,
          }
        });
        
        const selfieData = processImageResponse(selfieResponse, "");
        if (!selfieData.image) {
            console.warn("[System] Selfie generation resulted in no image (likely filtered or model error).");
        }
        return selfieData.image;

    } catch (err) {
        console.error("Selfie generation failed:", err);
        return undefined;
    }
};

/**
 * Main entry point for communicating with Eve.
 * Routes between Chat (Text/Vision) and Flash Image (Generation/Editing).
 * Accepts 'history' to restore context if the session was lost (e.g. HMR or reload).
 * Support streaming via onStream callback.
 */
export const sendMessageToEve = async (
  message: string, 
  tier: ModelTier, 
  history: Message[],
  attachmentBase64?: string,
  forceImageGeneration: boolean = false,
  apiKey?: string,
  onStream?: (text: string) => void
): Promise<{ text: string; image?: string; visualPrompt?: string }> => {
  const key = apiKey || process.env.API_KEY || '';
  const ai = new GoogleGenAI({ apiKey: key });

  // Auto-init/Restore if missing or tier changed
  if (!chatSession || currentTier !== tier) {
    console.log("Restoring session before sending message...");
    await startChatWithHistory(tier, history, apiKey);
  }

  const mimeType = attachmentBase64 ? getMimeType(attachmentBase64) : 'image/jpeg';
  const cleanBase64 = attachmentBase64 ? attachmentBase64.replace(/^data:image\/\w+;base64,/, "") : null;
  const imageModel = MODELS[tier].image;

  try {
    // ROUTE 1: Image Editing (User provides image + edit intent OR forced mode)
    if (attachmentBase64 && (isImageEditingIntent(message) || forceImageGeneration)) {
      const response = await ai.models.generateContent({
        model: imageModel,
        contents: {
          parts: [
            { inlineData: { data: cleanBase64!, mimeType } },
            { text: message }
          ]
        },
        config: {
            imageConfig: { aspectRatio: '9:16' },
            safetySettings: SAFETY_SETTINGS,
        }
      });
      return processImageResponse(response, "I've evolved the visual based on your request.");
    }

    // ROUTE 2: Image Generation (No Attachment + (Gen Intent OR Forced Mode))
    if (!attachmentBase64 && (isImageGenerationIntent(message) || forceImageGeneration)) {
      // If user asks for "Selfie" in generation mode, append Eve's description
      let prompt = message;
      if (message.toLowerCase().includes('selfie')) {
         prompt = `Create a photorealistic selfie of this fictional character: ${EVE_APPEARANCE}. Action: ${message}`;
      }
      
      const response = await ai.models.generateContent({
        model: imageModel,
        contents: {
          parts: [{ text: prompt }]
        },
        config: {
            imageConfig: { aspectRatio: '9:16' },
            safetySettings: SAFETY_SETTINGS,
        }
      });
      return processImageResponse(response, "Here is what I visualized for you.");
    }

    // ROUTE 3: Standard Chat / Vision (History aware)
    let msgContent: any = message;
    if (attachmentBase64) {
      msgContent = {
        parts: [
          { inlineData: { data: cleanBase64!, mimeType } },
          { text: message }
        ]
      };
    }

    // Safety check: if session is still null (rare init failure), try one last force init
    if (!chatSession) {
         initializeChat(tier, [], apiKey); 
    }

    let replyText = "";

    // USE STREAMING IF CALLBACK PROVIDED
    if (onStream) {
      const streamResult = await chatSession!.sendMessageStream({ message: msgContent });
      
      for await (const chunk of streamResult) {
        const chunkText = chunk.text;
        if (chunkText) {
          replyText += chunkText;
          onStream(replyText); // Notify UI with accumulated text
        }
      }
    } else {
      // Fallback to standard await
      const result: GenerateContentResponse = await chatSession!.sendMessage({ message: msgContent });
      replyText = result.text || "";
    }

    // --- VISUAL TRIGGER PROTOCOL HANDLER ---
    // Regex to match [SELFIE] or [SELFIE: description]
    // Captures the description group if present
    const selfieMatch = replyText.match(/\[SELFIE(?::\s*(.*?))?\]/);
    let visualPrompt: string | undefined;

    if (selfieMatch) {
      console.log("[System] Visual Trigger Protocol Activated (Async)");
      visualPrompt = selfieMatch[1] || "looking at the camera"; // Captured description or default
      
      // Clean the tag from the text
      replyText = replyText.replace(/\[SELFIE(?::\s*.*?)?\]/g, "").trim();
    }

    return { text: replyText, visualPrompt };

  } catch (error) {
    console.error("Error communicating with Eve:", error);
    throw error;
  }
};

export const startChatWithHistory = async (tier: ModelTier, history: Message[], apiKey?: string) => {
  if (!history || history.length === 0) {
    initializeChat(tier, [], apiKey);
    return;
  }

  try {
    const geminiHistory: any[] = [];
    
    // Flatten logic with Synthetic Turns
    for (const h of history) {
      if (h.isError) continue;

      // 1. Handle Role: USER
      if (h.role === 'user') {
         const parts: any[] = [];
         // Add Image if present
         if (h.image) {
           try {
              const mimeType = getMimeType(h.image);
              const data = h.image.replace(/^data:image\/\w+;base64,/, "");
              if (data && mimeType) {
                 parts.push({ inlineData: { mimeType, data } });
              }
           } catch(e) { console.warn("Skipping invalid image in user history"); }
         }
         // Add Text
         if (h.text && h.text.trim() !== "") {
           parts.push({ text: h.text });
         }
         if (parts.length > 0) {
            geminiHistory.push({ role: 'user', parts });
         }
      } 
      // 2. Handle Role: MODEL
      else if (h.role === 'model') {
         // Add Text content to Model turn
         const textParts = [{ text: h.text || "..." }];
         geminiHistory.push({ role: 'model', parts: textParts });

         // CRITICAL: Inject Model-Generated Image as a SYNTHETIC USER TURN
         // The API does not support images in 'model' history.
         // We feed it back as a 'user' context so the model can "see" what it generated.
         if (h.image) {
            try {
              const mimeType = getMimeType(h.image);
              const data = h.image.replace(/^data:image\/\w+;base64,/, "");
              if (data && mimeType) {
                 geminiHistory.push({ 
                    role: 'user', 
                    parts: [
                      { inlineData: { mimeType, data } },
                      { text: "[System: This is the visual content generated in the previous turn]" }
                    ]
                 });
              }
            } catch(e) { console.warn("Skipping invalid image in model history"); }
         }
      }
    }

    // 3. MERGE PASS: Consolidate adjacent same-role turns
    // (User -> Synthetic User -> User) should become one big User turn if possible,
    // or we rely on the Alternation Fix below.
    // Actually, simple merging is safer for Gemini API.
    const mergedHistory: any[] = [];
    if (geminiHistory.length > 0) {
        let currentTurn = geminiHistory[0];
        
        for (let i = 1; i < geminiHistory.length; i++) {
            const nextTurn = geminiHistory[i];
            if (nextTurn.role === currentTurn.role) {
                // Merge parts
                currentTurn.parts.push(...nextTurn.parts);
            } else {
                mergedHistory.push(currentTurn);
                currentTurn = nextTurn;
            }
        }
        mergedHistory.push(currentTurn);
    }

    // 4. CRITICAL FIX: Ensure history starts with a 'user' message.
    while (mergedHistory.length > 0 && mergedHistory[0].role === 'model') {
      mergedHistory.shift();
    }

    // 5. Ensure alternating turns (Trailing User -> Dummy Model)
    if (mergedHistory.length > 0 && mergedHistory[mergedHistory.length - 1].role === 'user') {
       mergedHistory.push({ role: 'model', parts: [{ text: "..." }] });
    }

    initializeChat(tier, mergedHistory, apiKey);
  } catch (e) {
    console.error("Failed to reconstruct history:", e);
    initializeChat(tier, [], apiKey);
  }
};

const processImageResponse = (response: GenerateContentResponse, fallbackText: string): { text: string, image?: string } => {
  let image: string | undefined;
  let text = "";

  if (response.candidates?.[0]?.content?.parts) {
    for (const part of response.candidates[0].content.parts) {
      if (part.inlineData) {
        image = `data:${part.inlineData.mimeType || 'image/png'};base64,${part.inlineData.data}`;
      } else if (part.text) {
        text += part.text;
      }
    }
  }

  return { 
    text: text || fallbackText, 
    image 
  };
};

const getMimeType = (dataUrl: string): string => {
  const match = dataUrl.match(/^data:(.*);base64,/);
  return match ? match[1] : 'image/jpeg';
};

// Helper to convert ArrayBuffer to Base64 (Browser compatible)
const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

// Deprecated direct export
export const editImageWithEve = async (base64Image: string, prompt: string, tier: ModelTier) => {
  const res = await sendMessageToEve(prompt, tier, [], base64Image, true);
  return res.image || null;
};
