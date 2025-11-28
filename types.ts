
export interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  image?: string; // Base64 string for images displayed in chat
  isError?: boolean;
  isImageLoading?: boolean;
}

export type ModelTier = 'free' | 'pro';

export interface EveConfig {
  voiceEnabled: boolean;
  personality: 'default' | 'bananafy';
}

export interface ImageEditState {
  originalImage: string | null; // Base64
  generatedImage: string | null; // Base64
  prompt: string;
  isLoading: boolean;
}

export interface ApiKeyDef {
  id: string;
  label: string;
  key: string;
}
