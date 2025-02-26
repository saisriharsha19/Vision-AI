"use client";

import { useState, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { motion } from "framer-motion";
import { 
  Copy, Trash2, Moon, Sun, Mic, Send, 
  Paperclip, Download, Search, Smile, Clock,
  Check, CheckCheck, UserCircle, BotIcon,
  Code, Link, ImageIcon, FileText, Upload,
  Trash, FileUp, BookOpen, Database, AlertCircle
} from "lucide-react";

type ContentType = "text" | "code" | "link" | "image" | "markdown" | "json";

type Message = {
  id: string;
  role: "user" | "bot";
  content: string;
  contentType?: ContentType;
  timestamp: Date;
  status: "sending" | "sent" | "delivered" | "read";
  reactions?: string[];
  attachments?: Array<{
    type: string;
    url: string;
    name: string;
    docId?: string; // Add document ID for uploaded RAG documents
  }>;
};

type Document = {
  id: string;
  filename: string;
  upload_date: string;
  doc_type: string;
};

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [darkMode, setDarkMode] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [botIsTyping, setBotIsTyping] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [showDocuments, setShowDocuments] = useState(false);
  const [isUploadingDoc, setIsUploadingDoc] = useState(false);
  const [uploadError, setUploadError] = useState("");
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const docInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  
  const API_URL = "http://localhost:8000";
  
  useEffect(() => {
    const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDarkMode(isDark);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);
  
  useEffect(() => {
    const savedMessages = localStorage.getItem("chatMessages");
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages);
        const messagesWithDateObjects = parsedMessages.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(messagesWithDateObjects);
      } catch (error) {
        console.error("Error parsing saved messages:", error);
      }
    }
  }, []);
  
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem("chatMessages", JSON.stringify(messages));
    }
  }, [messages]);
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  useEffect(() => {
    if (showDocuments) {
      fetchDocuments();
    }
  }, [showDocuments]);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const generateId = () => {
    return Math.random().toString(36).substring(2, 15);
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      } else {
        console.error("Failed to fetch documents:", await res.text());
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  };
  
  const uploadDocument = async (file: File) => {
    setIsUploadingDoc(true);
    setUploadError("");
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const res = await fetch(`${API_URL}/upload_document`, {
        method: "POST",
        body: formData,
      });
      
      if (res.ok) {
        const data = await res.json();
        setDocuments(prev => [...prev, data]);
        
        const systemMessage: Message = {
          id: generateId(),
          role: "bot",
          content: `Document "${data.filename}" uploaded successfully and is now available for answering queries.`,
          timestamp: new Date(),
          status: "read"
        };
        
        setMessages(prev => [...prev, systemMessage]);
        return data;
      } else {
        const errorText = await res.text();
        setUploadError(`Failed to upload document: ${errorText}`);
        console.error("Failed to upload document:", errorText);
        return null;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      setUploadError(`Error uploading document: ${errorMessage}`);
      console.error("Error uploading document:", error);
      return null;
    } finally {
      setIsUploadingDoc(false);
    }
  };
  
  const deleteDocument = async (documentId: string) => {
    try {
      const res = await fetch(`${API_URL}/documents/${documentId}`, {
        method: "DELETE"
      });
      
      if (res.ok) {
        setDocuments(prev => prev.filter(doc => doc.id !== documentId));
        
        const systemMessage: Message = {
          id: generateId(),
          role: "bot",
          content: "Document removed successfully.",
          timestamp: new Date(),
          status: "read"
        };
        
        setMessages(prev => [...prev, systemMessage]);
      } else {
        console.error("Failed to delete document:", await res.text());
      }
    } catch (error) {
      console.error("Error deleting document:", error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() && selectedFiles.length === 0) return;
  
    let documentUploaded = false;
    let uploadedDocAttachments: Array<{type: string, url: string, name: string, docId?: string}> = [];
    
    if (selectedFiles.length > 0) {
      for (const file of selectedFiles) {
        // Document file extensions
        const docExtensions = ['pdf', 'txt', 'doc', 'docx', 'csv', 'md'];
        const fileExt = file.name.split('.').pop()?.toLowerCase() || '';
        
        if (docExtensions.includes(fileExt)) {
          // This is a document - upload to RAG system
          const uploadResult = await uploadDocument(file);
          if (uploadResult) {
            documentUploaded = true;
            uploadedDocAttachments.push({
              type: file.type,
              url: URL.createObjectURL(file), // Just for display
              name: file.name,
              docId: uploadResult.id
            });
          }
        }
      }
    }
    
    const regularAttachments = await Promise.all(
      selectedFiles
        .filter(file => {
          const fileExt = file.name.split('.').pop()?.toLowerCase() || '';
          const docExtensions = ['pdf', 'txt', 'doc', 'docx', 'csv', 'md'];
          return !docExtensions.includes(fileExt);
        })
        .map(async (file) => {
          const url = URL.createObjectURL(file);
          return {
            type: file.type,
            url: url,
            name: file.name
          };
        })
    );
    
    const allAttachments = [...uploadedDocAttachments, ...regularAttachments];
    
    if (documentUploaded && !input.trim() && regularAttachments.length === 0) {
      setSelectedFiles([]);
      return;
    }
  
    const contentType = detectContentType(input.trim());
    
    const newMessage: Message = {
      id: generateId(),
      role: "user",
      content: input.trim(),
      contentType: contentType,
      timestamp: new Date(),
      status: "sending",
      attachments: allAttachments.length > 0 ? allAttachments : undefined
    };
    
    const newMessages = [...messages, newMessage];
    setMessages(newMessages);
    setInput("");
    setSelectedFiles([]);
    
    setTimeout(() => {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id ? { ...msg, status: "sent" } : msg
        )
      );
    }, 500);

    setBotIsTyping(true);
    
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: input,
          attachments: allAttachments.map(a => a.name)
        }),
      });
      
      setMessages(prev => 
        prev.map(msg => 
          msg.id === newMessage.id ? { ...msg, status: "delivered" } : msg
        )
      );
      
      const data = await res.json();
      
      setBotIsTyping(false);
      
      // Add bot response
      const botMessage: Message = {
        id: generateId(),
        role: "bot",
        content: data.response,
        contentType: detectContentType(data.response),
        timestamp: new Date(),
        status: "read"
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      setTimeout(() => {
        setMessages(prev => 
          prev.map(msg => 
            msg.id === newMessage.id ? { ...msg, status: "read" } : msg
          )
        );
      }, 1000);
      
    } catch (error) {
      console.error("Error fetching response:", error);
      setBotIsTyping(false);
      
      const errorMessage: Message = {
        id: generateId(),
        role: "bot",
        content: "Sorry, I couldn't process your request. Please try again later.",
        timestamp: new Date(),
        status: "read"
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    localStorage.removeItem("chatMessages");
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };
  
  const addReaction = (messageId: string, emoji: string) => {
    setMessages(prev => 
      prev.map(msg => {
        if (msg.id === messageId) {
          const reactions = msg.reactions || [];
          const updatedReactions = reactions.includes(emoji) 
            ? reactions.filter(r => r !== emoji)
            : [...reactions, emoji];
          
          return { ...msg, reactions: updatedReactions };
        }
        return msg;
      })
    );
  };
  
  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };
  
  const handleDocSelect = () => {
    docInputRef.current?.click();
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };
  
  const handleDocFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadDocument(file);
      
      if (e.target instanceof HTMLInputElement) {
        e.target.value = '';
      }
    }
  };
  



  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];
        
        mediaRecorder.ondataavailable = (e) => {
          audioChunksRef.current.push(e.data);
        };
        
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
          
          setIsRecording(true);
          
          try {
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.wav');
            
            // Add any additional metadata if needed
            const requestData = {
              filename: 'recording.wav',
            };
            formData.append('request_data', JSON.stringify(requestData));
            
            // Send to the backend API
            const response = await fetch(`${API_URL}/transcribe_audio`, {
              method: 'POST',
              body: formData,
            });
            
            if (!response.ok) {
              throw new Error(`Server responded with ${response.status}`);
            }
            
            const result = await response.json();
            
            setInput(prev => prev + (prev ? ' ' : '') + result.text);
          } catch (error) {
            console.error("Error transcribing audio:", error);
            setInput(prev => prev + (prev ? ' ' : '') + "[Voice transcription failed]");
          } finally {
            setIsRecording(false);
            
            stream.getTracks().forEach(track => track.stop());
          }
        };
        
        mediaRecorder.start();
        setIsRecording(true);
      } catch (error) {
        console.error("Error accessing microphone:", error);
      }
    } else {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
        setIsRecording(false);
      }
    }
  };
  
  const toggleSearch = () => {
    setIsSearching(!isSearching);
    if (!isSearching) {
      setSearchTerm("");
    }
  };
  
  const toggleDocuments = () => {
    setShowDocuments(!showDocuments);
  };
  
  const exportChat = () => {
    // Create a text version of the chat
    const chatText = messages.map(msg => 
      `[${msg.timestamp.toLocaleString()}] ${msg.role === 'user' ? 'You' : 'Bot'}: ${msg.content}`
    ).join('\n\n');
    
    // Create a download link
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  const detectContentType = (content: string): ContentType => {
    // Check if it's code (basic heuristic: contains code blocks or common syntax)
    if (
      content.includes('```') || 
      content.includes('function ') || 
      content.includes('const ') || 
      content.includes('class ') ||
      content.includes('import ') ||
      content.includes('export ') ||
      content.includes(';') && content.includes('{') && content.includes('}')
    ) {
      return 'code';
    }
    
    // Check if it's JSON
    if (
      (content.startsWith('{') && content.endsWith('}')) ||
      (content.startsWith('[') && content.endsWith(']'))
    ) {
      try {
        JSON.parse(content);
        return 'json';
      } catch (e) {
        // Not valid JSON
      }
    }
    
    // Check if it's an image reference
    if (
      content.match(/\.(jpeg|jpg|gif|png|svg)$/i) ||
      content.startsWith('data:image/')
    ) {
      return 'image';
    }
    
    // Check if it's a link
    const urlPattern = /https?:\/\/[^\s]+/;
    if (urlPattern.test(content)) {
      return 'link';
    }
    
    // Check if it's markdown
    if (
      content.includes('# ') ||
      content.includes('## ') ||
      content.includes('**') ||
      content.includes('__') ||
      content.includes('- ') ||
      content.includes('> ') ||
      content.includes('![') ||
      content.includes('[') && content.includes('](')
    ) {
      return 'markdown';
    }
    
    // Default to regular text
    return 'text';
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };
  
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };
  
  const renderMessageStatus = (status: Message['status']) => {
    switch (status) {
      case 'sending':
        return <Clock className="w-3 h-3 text-gray-400" />;
      case 'sent':
        return <Check className="w-3 h-3 text-gray-400" />;
      case 'delivered':
        return <Check className="w-3 h-3 text-blue-500" />;
      case 'read':
        return <CheckCheck className="w-3 h-3 text-blue-500" />;
    }
  };
  
  const filteredMessages = searchTerm.trim() 
    ? messages.filter(msg => 
        msg.content.toLowerCase().includes(searchTerm.toLowerCase())
      )
    : messages;

return (
  <div className="flex flex-col items-center justify-center min-h-screen w-full bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-2 sm:p-4">
    {/* Container with max width for large screens but responsive on small screens */}
    <div className="w-full max-w-3xl mx-auto flex flex-col h-[98vh]">
      {/* Header Section with Controls */}
      <div className="w-full mb-2 sm:mb-4 flex justify-between items-center p-2 sm:p-3 bg-white dark:bg-gray-800 rounded-2xl shadow-md">
        <h1 className="text-lg sm:text-xl font-bold text-gray-800 dark:text-white flex items-center">
          <BotIcon className="w-5 h-5 mr-2 text-blue-500" />
          Vision AI Chatbot
        </h1>
        
        <div className="flex gap-1 sm:gap-2">
          <button
            onClick={toggleSearch}
            className={`p-1.5 sm:p-2 rounded-full ${isSearching ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'} hover:bg-opacity-90 transition`}
            title="Search messages"
          >
            <Search className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
          
          <button
            onClick={toggleDocuments}
            className={`p-1.5 sm:p-2 rounded-full ${showDocuments ? 'bg-blue-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'} hover:bg-opacity-90 transition`}
            title="Manage documents"
          >
            <Database className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
          
          <button
            onClick={exportChat}
            className="bg-gray-100 dark:bg-gray-700 p-1.5 sm:p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition text-gray-700 dark:text-gray-200"
            title="Export chat"
          >
            <Download className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
          
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="bg-gray-100 dark:bg-gray-700 p-1.5 sm:p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition"
            title="Toggle dark mode"
          >
            {darkMode ? 
              <Sun className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-500" /> : 
              <Moon className="w-4 h-4 sm:w-5 sm:h-5 text-gray-700" />
            }
          </button>
        </div>
      </div>
      
      {/* Search Bar (conditional) */}
      {isSearching && (
        <div className="w-full mb-2 sm:mb-4">
          <Input
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search messages..."
            className="w-full p-2 sm:p-3 border rounded-xl bg-white dark:bg-gray-800 dark:text-white shadow-md"
          />
        </div>
      )}
      
      {/* Document Management Panel (conditional) */}
      {showDocuments && (
        <Card className="w-full mb-2 sm:mb-4 shadow-lg rounded-2xl  bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 flex-grow flex flex-col">
          <CardContent className="p-2 sm:p-4 scrollbar-thin">
            <div className="flex justify-between items-center mb-2 sm:mb-4">
              <h2 className="text-base sm:text-lg font-semibold dark:text-white flex items-center">
                <Database className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-blue-500" />
                Knowledge Base
              </h2>
              <Button 
                onClick={handleDocSelect}
                className="bg-blue-600 hover:bg-blue-700 text-white px-2 sm:px-3 py-1 sm:py-2 rounded-lg flex items-center gap-1 sm:gap-2 text-xs sm:text-sm"
                disabled={isUploadingDoc}
              >
                {isUploadingDoc ? (
                  <div className="animate-spin h-3 w-3 sm:h-4 sm:w-4 border-2 border-white border-t-transparent rounded-full" />
                ) : (
                  <FileUp className="w-3 h-3 sm:w-4 sm:h-4" />
                )}
                Upload
              </Button>
            </div>
            
            {uploadError && (
              <div className="mb-2 sm:mb-4 p-2 sm:p-3 bg-red-100 border border-red-400 text-red-700 rounded-md flex items-start">
                <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 mr-2 flex-shrink-0 mt-0.5" />
                <p className="text-xs sm:text-sm">{uploadError}</p>
              </div>
            )}
            
            {documents.length === 0 ? (
              <div className="text-center py-4 sm:py-6 text-gray-500 dark:text-gray-400">
                <BookOpen className="w-8 h-8 sm:w-10 sm:h-10 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No documents uploaded yet.</p>
                <p className="text-xs mt-1">Upload documents to enhance responses.</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-40 sm:max-h-48 overflow-y-auto pr-1 scrollbar-thin">
                {documents.map(doc => (
                  <div 
                    key={doc.id} 
                    className="p-2 bg-gray-50 dark:bg-gray-700 rounded-lg flex items-center justify-between"
                  >
                    <div className="flex items-center overflow-hidden">
                      <FileText className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-blue-500 flex-shrink-0" />
                      <div className="overflow-hidden">
                        <p className="font-medium text-xs sm:text-sm dark:text-white truncate">{doc.filename}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {new Date(doc.upload_date).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => deleteDocument(doc.id)}
                      className="p-1 text-red-500 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 flex-shrink-0"
                      title="Delete document"
                    >
                      <Trash className="w-3 h-3 sm:w-4 sm:h-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Main Chat Area */}
      <Card className="w-full shadow-lg rounded-2xl overflow-hidden bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 flex-grow flex flex-col">
        <CardContent className="flex-grow overflow-y-auto p-2 sm:p-4 flex flex-col gap-2 sm:gap-3 scrollbar-thin">
          {/* Initial welcome message if no messages */}
          {messages.length === 0 && (
            <div className="text-center py-6 sm:py-10 flex-grow flex flex-col items-center justify-center">
              <BotIcon className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-3 sm:mb-4 text-blue-500 opacity-70" />
              <h2 className="text-lg sm:text-xl font-semibold mb-2 dark:text-white">Welcome to Vision AI Chatbot</h2>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 max-w-md">
                I can answer questions based on your uploaded documents. Try uploading some documents first!
              </p>
              <Button 
                onClick={toggleDocuments}
                className="bg-blue-600 hover:bg-blue-700 text-white px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg flex items-center gap-2 text-sm"
              >
                <Database className="w-4 h-4 sm:w-5 sm:h-5" />
                Manage Documents
              </Button>
            </div>
          )}
          
          {filteredMessages.map((msg, i) => (
            <div key={msg.id} className={`relative ${searchTerm && msg.content.toLowerCase().includes(searchTerm.toLowerCase()) ? 'bg-yellow-50 dark:bg-yellow-900/30 p-1 rounded-xl' : ''}`}>
              <div className="flex items-start">
                {/* Avatar */}
                <div className="mr-2 flex-shrink-0">
                  {msg.role === "user" ? (
                    <div className="w-6 h-6 sm:w-8 sm:h-8 bg-blue-500 rounded-full flex items-center justify-center text-white">
                      <UserCircle className="w-4 h-4 sm:w-5 sm:h-5" />
                    </div>
                  ) : (
                    <div className="w-6 h-6 sm:w-8 sm:h-8 bg-green-500 rounded-full flex items-center justify-center text-white">
                      <BotIcon className="w-4 h-4 sm:w-5 sm:h-5" />
                    </div>
                  )}
                </div>
                
                <div className="flex-grow">
                  {/* Message with timestamp */}
                  <div className="flex items-center mb-1">
                    <span className="font-medium text-xs sm:text-sm text-gray-700 dark:text-gray-300">
                      {msg.role === "user" ? "You" : "Bot"}
                    </span>
                    <span className="text-xs text-gray-500 ml-2">
                      {formatTimestamp(msg.timestamp)}
                    </span>
                    
                    {/* Content type indicator */}
                    {msg.contentType && msg.contentType !== 'text' && (
                      <span className="ml-2 hidden sm:flex items-center text-xs text-gray-500 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                        {msg.contentType === 'code' && <Code className="w-3 h-3 mr-1" />}
                        {msg.contentType === 'json' && <Code className="w-3 h-3 mr-1" />}
                        {msg.contentType === 'link' && <Link className="w-3 h-3 mr-1" />}
                        {msg.contentType === 'image' && <ImageIcon className="w-3 h-3 mr-1" />}
                        {msg.contentType === 'markdown' && <FileText className="w-3 h-3 mr-1" />}
                        {msg.contentType.charAt(0).toUpperCase() + msg.contentType.slice(1)}
                      </span>
                    )}
                    
                    {/* Message status (only for user messages) */}
                    {msg.role === "user" && (
                      <span className="ml-auto">
                        {renderMessageStatus(msg.status)}
                      </span>
                    )}
                  </div>
                  
                  {/* Message bubble */}
                  <motion.div
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    className={`p-2 sm:p-3 rounded-2xl break-words text-sm sm:text-base ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200"
                    }`}
                  >
                    {/* Content based on type */}
                    {msg.contentType === 'code' ? (
                      <div className="font-mono text-xs sm:text-sm bg-gray-800 dark:bg-gray-900 text-green-400 p-2 rounded overflow-x-auto">
                        <pre>{msg.content}</pre>
                      </div>
                    ) : msg.contentType === 'json' ? (
                      <div className="font-mono text-xs sm:text-sm bg-gray-800 dark:bg-gray-900 text-yellow-300 p-2 rounded overflow-x-auto">
                        <pre>{msg.content}</pre>
                      </div>
                    ) : msg.contentType === 'link' ? (
                      <div>
                        <a 
                          href={msg.content} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-blue-300 dark:text-blue-400 underline break-all"
                        >
                          {msg.content}
                        </a>
                      </div>
                    ) : msg.contentType === 'image' ? (
                      <div>
                        <div className="mt-1">
                          <img 
                            src={msg.content} 
                            alt="Shared image" 
                            className="max-w-full rounded"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.onerror = null;
                              target.src = '/placeholder-image.jpg';
                            }}
                          />
                        </div>
                      </div>
                    ) : msg.contentType === 'markdown' ? (
                      <div className="prose dark:prose-invert prose-sm max-w-none">
                        {/* In a real app, you'd use a markdown parser here */}
                        {msg.content}
                      </div>
                    ) : (
                      // Default text
                      <div>{msg.content}</div>
                    )}
                    
                    {/* Attachments if any */}
                    {msg.attachments && msg.attachments.length > 0 && (
                      <div className="mt-2 space-y-1 sm:space-y-2">
                        {msg.attachments.map((attachment, idx) => (
                          <div key={idx} className="flex items-center p-1.5 sm:p-2 rounded bg-white dark:bg-gray-600 bg-opacity-20 text-xs sm:text-sm">
                            {attachment.docId ? (
                              // RAG document indicator
                              <Database className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2 text-green-500" />
                            ) : (
                              <Paperclip className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-2" />
                            )}
                            <span className="truncate">
                              {attachment.name}
                              {attachment.docId && <span className="ml-1 text-xs text-gray-300">(KB)</span>}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </motion.div>
                  
                  {/* Reactions */}
                  {msg.reactions && msg.reactions.length > 0 && (
                    <div className="flex mt-1 space-x-1">
                      {msg.reactions.map((emoji, idx) => (
                        <span 
                          key={idx} 
                          className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded-full text-xs sm:text-sm"
                        >
                          {emoji}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                
                {/* Message Actions - Show on hover or tap on mobile */}
                <div className="hidden group-hover:flex sm:flex flex-row sm:flex-col ml-1 sm:ml-2 sm:space-y-1 space-x-1 sm:space-x-0">
                  <button
                    onClick={() => copyToClipboard(msg.content)}
                    className="p-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                    title="Copy message"
                  >
                    <Copy className="w-3 h-3 sm:w-4 sm:h-4" />
                  </button>
                  
                  <button
                    onClick={() => addReaction(msg.id, "👍")}
                    className="p-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition"
                    title="Add reaction"
                  >
                    <Smile className="w-3 h-3 sm:w-4 sm:h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
          
          {/* Bot typing indicator */}
          {botIsTyping && (
            <div className="flex items-start">
              <div className="mr-2">
                <div className="w-6 h-6 sm:w-8 sm:h-8 bg-green-500 rounded-full flex items-center justify-center text-white">
                  <BotIcon className="w-4 h-4 sm:w-5 sm:h-5" />
                </div>
              </div>
              <motion.div 
                className="p-2 sm:p-3 rounded-xl bg-gray-100 dark:bg-gray-700"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                <div className="flex space-x-1">
                  <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-gray-500 dark:bg-gray-300"></div>
                  <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-gray-500 dark:bg-gray-300"></div>
                  <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-gray-500 dark:bg-gray-300"></div>
                </div>
              </motion.div>
            </div>
          )}
          
          {/* This empty div helps us scroll to bottom */}
          <div ref={messagesEndRef} />
        </CardContent>

        {/* Input Area - Always visible at bottom */}
        <div className="sticky bottom-0 bg-white dark:bg-gray-800 p-2 sm:p-4 border-t dark:border-gray-700">
          {/* Selected Files Preview */}
          {selectedFiles.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1 sm:gap-2">
              {selectedFiles.map((file, idx) => {
                // Check if it's a document that will be processed by RAG
                const fileExt = file.name.split('.').pop()?.toLowerCase() || '';
                const docExtensions = ['pdf', 'txt', 'doc', 'docx', 'csv', 'md'];
                const isRagDocument = docExtensions.includes(fileExt);
                
                return (
                  <div 
                    key={idx} 
                    className={`px-1.5 sm:px-2 py-0.5 sm:py-1 ${isRagDocument ? 'bg-green-100 dark:bg-green-900' : 'bg-blue-100 dark:bg-blue-900'} rounded-lg text-xs flex items-center`}
                  >
                    {isRagDocument ? (
                      <Database className="w-2.5 h-2.5 sm:w-3 sm:h-3 mr-1 text-green-700 dark:text-green-500" />
                    ) : (
                      <Paperclip className="w-2.5 h-2.5 sm:w-3 sm:h-3 mr-1" />
                    )}
                    <span className="truncate max-w-[100px] sm:max-w-[150px]">{file.name}</span>
                    <span className="ml-1 text-xs opacity-70 hidden sm:inline">
                      {formatFileSize(file.size)}
                    </span>
                    <button 
                      onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== idx))}
                      className="ml-1 text-red-500 hover:text-red-700"
                    >
                      ×
                    </button>
                  </div>
                );
              })}
            </div>
          )}
          
          <div className="flex gap-1 sm:gap-2">
          <div className="relative w-full max-w-2xl mx-auto">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question..."
              className="w-full p-3 pr-12 border rounded-xl bg-gray-50 dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
            <button 
              onClick={sendMessage} 
              className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center justify-center w-8 h-8"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>

            {/* Mobile-optimized button row with smaller icons on small screens */}
            <div className="flex gap-1 sm:gap-2">
              <button
                onClick={handleFileSelect}
                className="p-2 sm:p-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
                title="Add attachment"
              >
                <Paperclip className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>
              
              <button
                onClick={toggleRecording}
                className={`p-2 sm:p-3 rounded-lg ${
                  isRecording 
                    ? "bg-red-500 text-white" 
                    : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600"
                }`}
                title={isRecording ? "Stop recording" : "Start voice recording"}
              >
                <Mic className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>

              {/* Only show clear chat on larger screens, move to menu on mobile */}
              <button 
                onClick={clearChat} 
                className="hidden sm:flex p-2 sm:p-3 bg-red-500 hover:bg-red-600 text-white rounded-lg items-center justify-center"
                title="Clear chat"
              >
                <Trash2 className="w-4 h-4 sm:w-5 sm:h-5" />
              </button>
            </div>
          </div>
          
          {/* Hidden file inputs */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            multiple
            className="hidden"
          />
          
          <input
            type="file"
            ref={docInputRef}
            onChange={handleDocFileChange}
            accept=".pdf,.txt,.doc,.docx,.csv,.md"
            className="hidden"
          />
        </div>
      </Card>
      
      {/* Mobile footer with version/help */}
      <div className="mt-2 text-center">
        <p className="text-xs text-gray-500 dark:text-gray-400">
        Vision AI Chatbot v2.0 <br /> Developed by Sai Sri harsha Guddati •
          <button className="ml-1 text-blue-500 hover:text-blue-600">
            Help
          </button>
        </p>
      </div>
    </div>
  </div>
);
}