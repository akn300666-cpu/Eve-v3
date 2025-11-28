import React, { useState, useRef, useEffect } from 'react';
import { editImageWithEve } from '../services/geminiService';
import { ModelTier } from '../types';

interface ImageEditorProps {
  onBack: () => void;
  modelTier: ModelTier;
}

const ImageEditor: React.FC<ImageEditorProps> = ({ onBack, modelTier }) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Handle paste events globally within the component
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (items) {
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf('image') !== -1) {
            const blob = items[i].getAsFile();
            if (blob) {
              const reader = new FileReader();
              reader.onload = (event) => {
                setSelectedImage(event.target?.result as string);
                setGeneratedImage(null);
              };
              reader.readAsDataURL(blob);
            }
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result as string);
        setGeneratedImage(null); // Reset previous generation
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
       const reader = new FileReader();
       reader.onloadend = () => {
         setSelectedImage(reader.result as string);
         setGeneratedImage(null);
       };
       reader.readAsDataURL(file);
    }
  };

  const handleEdit = async () => {
    if (!selectedImage || !prompt) return;

    setIsProcessing(true);
    try {
      const result = await editImageWithEve(selectedImage, prompt, modelTier);
      if (result) {
        setGeneratedImage(result);
      } else {
        alert("EVE couldn't visually edit the image. The model might have analyzed it instead of returning an image. Try a more direct instruction like 'Add a filter'.");
      }
    } catch (error) {
      console.error("Editing failed", error);
      alert("Something went wrong during the evolution process. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-full p-4 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <button 
          onClick={onBack}
          className="text-slate-400 hover:text-fuchsia-400 transition-colors flex items-center gap-2 text-sm uppercase tracking-widest"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd" />
          </svg>
          Return to Chat
        </button>
        <h2 className="text-lg font-serif italic text-fuchsia-500">Visual Evolution</h2>
      </div>

      <div className="flex-1 flex flex-col md:flex-row gap-6 overflow-y-auto">
        {/* Input Section */}
        <div className="flex-1 flex flex-col gap-4">
          <div 
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all h-64 relative overflow-hidden
              ${selectedImage ? 'border-fuchsia-500/30 bg-slate-800/30' : 'border-slate-700 hover:border-slate-500 hover:bg-slate-800/50'}
              ${isDragOver ? 'border-fuchsia-500 bg-fuchsia-500/10' : ''}
            `}
          >
            {selectedImage ? (
              <img src={selectedImage} alt="Original" className="h-full w-full object-contain rounded-lg relative z-10" />
            ) : (
              <div className="text-slate-500 relative z-10 pointer-events-none">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto mb-2 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p>Drop image here, paste (Ctrl+V), or click to upload</p>
                <p className="text-xs mt-1 text-slate-600">JPG, PNG supported</p>
              </div>
            )}
            <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-xs uppercase tracking-widest text-slate-500 font-bold">Evolution Directive</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="E.g., 'Add a retro filter', 'Remove the person in the background', 'Make it cyberpunk'"
              className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-sm focus:outline-none focus:border-fuchsia-500/50 resize-none h-24 placeholder-slate-600 transition-colors"
            />
          </div>

          <button
            onClick={handleEdit}
            disabled={!selectedImage || !prompt || isProcessing}
            className={`py-3 px-6 rounded-lg font-medium text-sm tracking-wide transition-all shadow-lg
              ${!selectedImage || !prompt || isProcessing 
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                : 'bg-gradient-to-r from-fuchsia-600 to-purple-600 text-white hover:shadow-fuchsia-500/20 active:scale-95'}
            `}
          >
            {isProcessing ? 'Evolving...' : 'Execute Edit'}
          </button>
        </div>

        {/* Output Section */}
        <div className="flex-1 border border-slate-800 rounded-xl bg-slate-900/50 p-4 flex flex-col">
           <div className="text-xs uppercase tracking-widest text-slate-500 font-bold mb-4">Result</div>
           <div className="flex-1 flex items-center justify-center rounded-lg bg-black/40 overflow-hidden relative min-h-[300px]">
             {isProcessing ? (
                <div className="flex flex-col items-center">
                  <div className="relative">
                    <div className="w-12 h-12 border-2 border-slate-700 rounded-full"></div>
                    <div className="absolute top-0 left-0 w-12 h-12 border-2 border-fuchsia-500 border-t-transparent rounded-full animate-spin"></div>
                  </div>
                  <span className="text-fuchsia-500/70 text-xs mt-4 animate-pulse">
                    {modelTier === 'pro' ? 'Gemini 3 Pro Processing...' : 'Gemini 2.5 Flash Processing...'}
                  </span>
                </div>
             ) : generatedImage ? (
                <img src={generatedImage} alt="Generated" className="max-w-full max-h-full object-contain animate-fade-in" />
             ) : (
               <div className="text-center p-8 opacity-40">
                  <div className="w-16 h-16 border-2 border-dashed border-slate-600 rounded-lg mx-auto mb-4"></div>
                  <span className="text-slate-500 text-sm italic">Edited image will appear here</span>
               </div>
             )}
           </div>
           {generatedImage && (
             <a href={generatedImage} download="eve-edit.png" className="mt-4 flex items-center justify-center gap-2 py-2 rounded-lg bg-slate-800 text-xs text-fuchsia-500 hover:text-fuchsia-400 hover:bg-slate-700 transition-colors">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
               </svg>
               Download Image
             </a>
           )}
        </div>
      </div>
    </div>
  );
};

export default ImageEditor;