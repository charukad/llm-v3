/**
 * CameraCapture Component
 * 
 * A component for capturing images of handwritten mathematical expressions
 * using the device's camera or by uploading image files.
 */

import React, { useState, useRef, useEffect } from 'react';
import './CameraCapture.css';

const CameraCapture = ({ 
  onCapture, 
  onCancel,
  aspectRatio = 4/3,
  maxWidth = 800,
  allowUpload = true 
}) => {
  // States
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraAvailable, setCameraAvailable] = useState(false);
  const [hasFlash, setHasFlash] = useState(false);
  const [flashEnabled, setFlashEnabled] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [error, setError] = useState(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [loading, setLoading] = useState(true);
  
  // References
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Initialize camera on mount
  useEffect(() => {
    checkCameraAvailability();
    
    return () => {
      // Clean up
      stopCamera();
    };
  }, []);
  
  // Check if camera is available
  const checkCameraAvailability = async () => {
    setLoading(true);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraAvailable(false);
      setError('Camera not supported by your browser');
      setLoading(false);
      return;
    }
    
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasCamera = devices.some(device => device.kind === 'videoinput');
      
      if (!hasCamera) {
        setCameraAvailable(false);
        setError('No camera detected on your device');
        setLoading(false);
        return;
      }
      
      setCameraAvailable(true);
      startCamera();
    } catch (err) {
      console.error('Error checking camera availability:', err);
      setCameraAvailable(false);
      setError('Failed to access camera: ' + err.message);
      setLoading(false);
    }
  };
  
  // Start camera
  const startCamera = async () => {
    if (!cameraAvailable) return;
    
    try {
      const constraints = {
        video: {
          facingMode: 'environment', // Use back camera on mobile devices
          aspectRatio,
          width: { max: maxWidth }
        }
      };
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Check if flash is available
      const tracks = stream.getVideoTracks();
      if (tracks.length > 0) {
        const capabilities = tracks[0].getCapabilities();
        setHasFlash(capabilities.torch || false);
      }
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error('Error starting camera:', err);
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermissionDenied(true);
        setError('Camera permission denied. Please allow camera access to use this feature.');
      } else {
        setError('Failed to start camera: ' + err.message);
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      const tracks = streamRef.current.getTracks();
      tracks.forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setCameraActive(false);
  };
  
  // Toggle flash
  const toggleFlash = async () => {
    if (!hasFlash || !streamRef.current) return;
    
    try {
      const tracks = streamRef.current.getVideoTracks();
      if (tracks.length > 0) {
        const track = tracks[0];
        const newFlashState = !flashEnabled;
        
        await track.applyConstraints({
          advanced: [{ torch: newFlashState }]
        });
        
        setFlashEnabled(newFlashState);
      }
    } catch (err) {
      console.error('Error toggling flash:', err);
    }
  };
  
  // Capture image
  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas dimensions to match video
    const { videoWidth, videoHeight } = video;
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    
    // Draw video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, videoWidth, videoHeight);
    
    // Get image as data URL
    const imageDataUrl = canvas.toDataURL('image/png');
    setCapturedImage(imageDataUrl);
    stopCamera();
  };
  
  // Retry capture
  const retryCapture = () => {
    setCapturedImage(null);
    startCamera();
  };
  
  // Use captured image
  const useImage = () => {
    if (!capturedImage || !onCapture) return;
    
    onCapture(capturedImage);
  };
  
  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Check file type
    if (!file.type.match('image.*')) {
      setError('Please select an image file');
      return;
    }
    
    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Image file size should be less than 10MB');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
      setCapturedImage(event.target.result);
      stopCamera();
    };
    
    reader.readAsDataURL(file);
  };
  
  // Trigger file input click
  const triggerFileUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  // Render
  if (loading) {
    return (
      <div className="camera-loading">
        <div className="loading-spinner"></div>
        <p>Initializing camera...</p>
      </div>
    );
  }
  
  if (permissionDenied) {
    return (
      <div className="camera-error-container">
        <div className="camera-error">
          <p>
            <span className="error-icon">⚠️</span>
            Camera permission denied
          </p>
          <p className="error-help">Please allow camera access in your browser settings to use this feature.</p>
        </div>
        
        {allowUpload && (
          <div className="camera-alternatives">
            <p>Alternatively, you can:</p>
            <button 
              type="button" 
              className="upload-button"
              onClick={triggerFileUpload}
            >
              Upload an Image
            </button>
            <input 
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="file-input"
            />
          </div>
        )}
        
        <button 
          type="button" 
          className="cancel-button"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    );
  }
  
  if (!cameraAvailable && !capturedImage) {
    return (
      <div className="camera-error-container">
        <div className="camera-error">
          <p>
            <span className="error-icon">⚠️</span>
            {error || 'Camera not available'}
          </p>
        </div>
        
        {allowUpload && (
          <div className="camera-alternatives">
            <p>Please use one of these alternatives:</p>
            <button 
              type="button" 
              className="upload-button"
              onClick={triggerFileUpload}
            >
              Upload an Image
            </button>
            <input 
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              className="file-input"
            />
          </div>
        )}
        
        <button 
          type="button" 
          className="cancel-button"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    );
  }
  
  return (
    <div className="camera-capture">
      {!capturedImage ? (
        <div className="camera-container">
          <div className="video-container">
            <video 
              ref={videoRef}
              autoPlay
              playsInline
              className="camera-video"
              onLoadedMetadata={() => setLoading(false)}
            />
            <div className="camera-guidelines">
              <div className="guideline-text">
                Position your handwritten math in this area
              </div>
            </div>
          </div>
          
          <div className="camera-controls">
            {hasFlash && (
              <button 
                type="button" 
                className={`flash-button ${flashEnabled ? 'active' : ''}`}
                onClick={toggleFlash}
                aria-label={flashEnabled ? 'Disable flash' : 'Enable flash'}
              >
                {flashEnabled ? '💡 Flash On' : '🔦 Flash Off'}
              </button>
            )}
            
            <button 
              type="button" 
              className="capture-button"
              onClick={captureImage}
              aria-label="Capture image"
            >
              <span className="capture-icon"></span>
            </button>
            
            {allowUpload && (
              <button 
                type="button" 
                className="upload-button-small"
                onClick={triggerFileUpload}
                aria-label="Upload image"
              >
                📁 Upload
              </button>
            )}
          </div>
          
          <canvas ref={canvasRef} className="hidden-canvas" />
          
          <input 
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            className="file-input"
          />
        </div>
      ) : (
        <div className="preview-container">
          <div className="image-preview">
            <img src={capturedImage} alt="Captured" className="preview-image" />
          </div>
          
          <div className="preview-controls">
            <button 
              type="button" 
              className="retry-button"
              onClick={retryCapture}
              aria-label="Retry capture"
            >
              Retry
            </button>
            
            <button 
              type="button" 
              className="use-image-button"
              onClick={useImage}
              aria-label="Use this image"
            >
              Use This Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
