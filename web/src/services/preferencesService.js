/**
 * Service for managing user preferences in the Mathematical Multimodal LLM system
 * Handles storing and retrieving user settings
 */

// Default preferences
const DEFAULT_PREFERENCES = {
  // Response preferences
  response: {
    showStepByStep: true,
    showVisualizations: true,
    detailLevel: 'medium', // 'basic', 'medium', 'detailed'
    preferredFormat: 'combined' // 'text', 'latex', 'combined'
  },
  
  // Display preferences
  display: {
    darkMode: false,
    highContrast: false,
    fontSize: 'medium', // 'small', 'medium', 'large'
    equationRenderer: 'mathjax' // 'mathjax', 'katex'
  },
  
  // Accessibility preferences
  accessibility: {
    screenReader: false,
    describeImages: true,
    keyboardNavigation: true,
    textToSpeech: false
  },
  
  // Input preferences
  input: {
    defaultInputMethod: 'text', // 'text', 'latex', 'drawing', 'camera'
    autoSuggest: true,
    latexCompletions: true,
    spellCheck: true
  }
};

// Storage key for preferences
const STORAGE_KEY = 'math_llm_preferences';

class PreferencesService {
  constructor() {
    this._preferences = this._loadPreferences();
  }
  
  // Load preferences from localStorage
  _loadPreferences() {
    try {
      const storedPreferences = localStorage.getItem(STORAGE_KEY);
      if (storedPreferences) {
        return { ...DEFAULT_PREFERENCES, ...JSON.parse(storedPreferences) };
      }
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
    
    return { ...DEFAULT_PREFERENCES };
  }
  
  // Save preferences to localStorage
  _savePreferences() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this._preferences));
    } catch (error) {
      console.error('Failed to save preferences:', error);
    }
  }
  
  // Get all preferences
  getAll() {
    return { ...this._preferences };
  }
  
  // Get a specific preference category
  get(category) {
    if (this._preferences[category]) {
      return { ...this._preferences[category] };
    }
    return null;
  }
  
  // Get a specific preference value
  getValue(category, key) {
    if (this._preferences[category] && this._preferences[category][key] !== undefined) {
      return this._preferences[category][key];
    }
    // Return default if available
    if (DEFAULT_PREFERENCES[category] && DEFAULT_PREFERENCES[category][key] !== undefined) {
      return DEFAULT_PREFERENCES[category][key];
    }
    return null;
  }
  
  // Update preferences for a category
  update(category, values) {
    if (this._preferences[category]) {
      this._preferences[category] = {
        ...this._preferences[category],
        ...values
      };
      this._savePreferences();
      return true;
    }
    return false;
  }
  
  // Update a single preference value
  setValue(category, key, value) {
    if (this._preferences[category]) {
      this._preferences[category][key] = value;
      this._savePreferences();
      return true;
    }
    return false;
  }
  
  // Reset preferences to defaults
  resetToDefaults() {
    this._preferences = { ...DEFAULT_PREFERENCES };
    this._savePreferences();
  }
  
  // Reset a specific category to defaults
  resetCategory(category) {
    if (DEFAULT_PREFERENCES[category]) {
      this._preferences[category] = { ...DEFAULT_PREFERENCES[category] };
      this._savePreferences();
      return true;
    }
    return false;
  }
  
  // Check if dark mode is enabled
  isDarkMode() {
    return this.getValue('display', 'darkMode') === true;
  }
  
  // Toggle dark mode
  toggleDarkMode() {
    const currentValue = this.getValue('display', 'darkMode');
    return this.setValue('display', 'darkMode', !currentValue);
  }
  
  // Get accessibility settings
  getAccessibilitySettings() {
    return this.get('accessibility');
  }
  
  // Get response format preferences
  getResponsePreferences() {
    return this.get('response');
  }
}

// Create a singleton instance
const preferencesService = new PreferencesService();
export default preferencesService;
