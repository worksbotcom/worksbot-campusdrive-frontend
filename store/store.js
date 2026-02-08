// store/store.js
"use client";
import { create } from "zustand";
import { startAssessment, submitAssessment, trackTabChange } from "@/utils/api";  
import { jwtDecode } from "jwt-decode";

const useStore = create((set) => ({
  user: null,
  token: null,
  error: null,
  loading: false,
  email: null,
  assessment: null,  // Store the current assessment
  assessmentId: null, // Store the assessment ID
  
  // Set email globally
  setEmail: (email) => set({ email }),

  // Set token and decode user information
  setToken: (token) => {
    try {
      sessionStorage.setItem("token", token);
      const decodedUser = jwtDecode(token);
      set({ token, user: decodedUser });
    } catch (error) {
      set({ token: null, user: null });
    }
  },

  // Clear session data
  clearSession: () => {
    sessionStorage.removeItem("token");
    set({ token: null, user: null, email: null, assessment: null, assessmentId: null });
  },

  // Initialize token from sessionStorage
  initializeToken: () => {
    if (typeof window !== "undefined") {
      const storedToken = sessionStorage.getItem("token");
      if (storedToken) {
        try {
          const decodedUser = jwtDecode(storedToken);
          set({ token: storedToken, user: decodedUser });
        } catch (error) {
          set({ token: null, user: null });
        }
      }
    }
  },

  // Set error globally
  setError: (error) => set({ error }),

  // Toggle loading globally
  setLoading: (loading) => set({ loading }),

  // Start an assessment
  startAssessment: async (role) => {
    set({ loading: true, error: null });
    try {
      const token = sessionStorage.getItem("token");
      if (!token) {
        throw new Error("No token found. Please sign in again.");
      }
      const assessment = await startAssessment(token, role);  // Call API to start assessment
      set({ assessment, assessmentId: assessment.id }); // Save the assessment ID in the state
    } catch (error) {
      set({ error: error.message });
      console.error(error);
    } finally {
      set({ loading: false });
    }
  },

  setAssessmentId: (assessmentId) => set({ assessmentId }),

  // Submit an assessment
  submitAssessment: async (score, questions) => {
    set({ loading: true, error: null });
    try {
      const token = sessionStorage.getItem("token");
      const assessmentId = get().assessmentId; // Get assessment ID from state
      if (!token || !assessmentId) {
        throw new Error("No token or assessment ID found. Please sign in again.");
      }
      const assessmentData = {
        assessmentId,
        score,
        questions,
      };
      await submitAssessment(token, assessmentData);  // Call API to submit assessment
    } catch (error) {
      set({ error: error.message });
      console.error(error);
    } finally {
      set({ loading: false });
    }
  },

  // Track tab change during assessment
  trackTabChange: async () => {
    try {
      const token = sessionStorage.getItem("token");
      const assessmentId = get().assessmentId; // Get assessment ID from state
      const userId = jwtDecode(token).id;  // Get user ID from the token
      await trackTabChange(token, userId, assessmentId);  // Call API to track tab change
    } catch (error) {
      console.log("Error tracking tab change:", error);
    }
  },

  // Handle logout
  logout: () => {
    set({ token: null, user: null, assessmentId: null });
    sessionStorage.removeItem("token");
    sessionStorage.removeItem("selectedRole");
    sessionStorage.removeItem("assessmentId");
  },
}));

export default useStore;
