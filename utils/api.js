// utils/api.js
import axios from "axios";

//const API_URL = "https://api.sourcesys.co/api";
const API_URL = "https://api1.worksbot.com/api";
//const API_URL = "http://localhost:5000/api";

// Register user
export const registerUser = async (formData) => {
  try {
    const response = await axios.post(`${API_URL}/auth/register`, formData);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Login user
export const loginUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/auth/login`, { email, password });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Generate OTP
export const generateOTP = async (email) => {
  try {
    const response = await axios.post(`${API_URL}/auth/generate-otp`, { email });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Verify OTP
export const verifyOTP = async (email, otp) => {
  try {
    const response = await axios.post(`${API_URL}/auth/verify-otp`, { email, otp });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Request password reset
export const requestResetPassword = async (email) => {
  try {
    const response = await axios.post(`${API_URL}/auth/request-password-reset`, { email });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Reset password
export const resetPassword = async (email, otp, newPassword) => {
  try {
    const response = await axios.post(`${API_URL}/auth/reset-password`, { email, otp, newPassword });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Start assessment
export const startAssessment = async (token, role) => {
  try {
    // Check if the user has already completed the assessment
    const checkResponse = await checkAssessmentStatus(token);
    if (checkResponse.data.completed) {
      return { status: "failed", message: "You have already completed the assessment", data: { completed: true } };
    }

    // If not completed, allow starting the assessment
    const response = await axios.post(
      `${API_URL}/assessment/start`, 
      { role }, 
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;  // Return assessment ID and other details
  } catch (error) {
    throw error.response ? error.response.data : error.message; // Add error handling for completed check
  }
};

// Submit assessment
export const submitAssessment = async (token, assessmentId, questionsData) => {
  try {
    const response = await axios.post(
      `${API_URL}/assessment/submit`,
      { assessmentId, questions: questionsData }, // Updated to match server expectation
      { headers: { Authorization: `Bearer ${token}` } }
    );

    return response.data;  // Return the response data
  } catch (error) {
    throw error.response ? error.response.data : error.message;
  }
};

// Track tab change during assessment
export const trackTabChange = async (token, userId, assessmentId) => {
  try {
    const response = await axios.post(
      `${API_URL}/assessment/track-tab-change`, 
      { userId, assessmentId }, 
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Check if the user has completed the assessment
export const checkAssessmentStatus = async (token) => {
  try {
    const response = await axios.get(`${API_URL}/assessment/check-status`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Fetch all users (Admin)
export const fetchPaginatedResults = async (token, params) => {
  try {
    const response = await axios.get(`${API_URL}/admin/users/results`, {
      headers: { Authorization: `Bearer ${token}` },
      params,
    });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const downloadResumes = async (token, params) => {
  // params = { collegeName, date }
  try {
    const response = await axios.get(`${API_URL}/admin/users/download-resumes`, {
      headers: { Authorization: `Bearer ${token}` },
      params,
      responseType: 'blob', // Important for file downloads
    });

    // Create a link to trigger the download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    const filename = `resumes-${params.collegeName || 'all'}.zip`;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const downloadFullCSV = async (token, params) => {
  try {
    const response = await axios.get(`${API_URL}/admin/download-csv`, 
      {
      headers: { Authorization: `Bearer ${token}` },
      params, // collegeName, startDate, endDate
      responseType: 'blob', // Important: tells axios to handle the response as a file
    });
    
    // Create a URL for the blob
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    
    // Extract filename from the Content-Disposition header sent by the backend
    const contentDisposition = response.headers['content-disposition'];
    let fileName = 'results.csv'; // Default filename
    if (contentDisposition) {
      const fileNameMatch = contentDisposition.match(/filename="(.+)"/);
      if (fileNameMatch && fileNameMatch.length === 2) {
        fileName = fileNameMatch[1];
      }
    }
    
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    
    // Clean up the object URL
    link.parentNode.removeChild(link);
    window.URL.revokeObjectURL(url);
    
  } catch (error) {
    console.error('Error downloading CSV:', error.response || error);
    // Rethrow to be caught by the component
    throw error.response ? error.response.data : error;
  }
};

// Search users by college name, email, phone, or role (Admin)
export const searchUsers = async (token, searchParams) => {
  try {
    const query = new URLSearchParams(searchParams).toString(); // Convert search params to query string
    const response = await axios.get(`${API_URL}/admin/users/search?${query}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data; // Return filtered users
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Fetch all assessments (Admin)
export const fetchAllAssessments = async (token) => {
  try {
    const response = await axios.get(`${API_URL}/admin/assessments/all`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data; // Return assessments with user and tab switch details
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};
// Fetch the current assessment toggle state
export const getToggleState = async (token) => {
  try {
    const response = await axios.get(`${API_URL}/toggle`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data; // { isActive: true/false }
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// Update the assessment toggle state (Admin only)
export const updateToggleState = async (token, isActive) => {
  try {
    const response = await axios.post(
      `${API_URL}/toggle`,
      { isActive },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data; // Updated toggle state
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};
