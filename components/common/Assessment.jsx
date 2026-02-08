"use client";
import { useEffect, useState } from "react";
import useStore from "@/store/store";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { startAssessment, submitAssessment, trackTabChange } from "@/utils/api";
import { assessmentRules } from "@/utils/data"; 
import { questions as allQuestions } from "@/utils/questions"; // Assuming your file is named questions.js
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { Federo } from 'next/font/google';
import Link from 'next/link';
import Logo from "./logo";

const federo = Federo({
  weight: '400',
  subsets: ['latin'],
  display: 'swap',
});

const Assessment = () => {
  const { user, token, initializeToken, logout } = useStore();
  const [loading, setLoading] = useState(true);
  const [isTermsAccepted, setIsTermsAccepted] = useState(false);
  const [isTermsCheckboxChecked, setIsTermsCheckboxChecked] = useState(false);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [timeLeft, setTimeLeft] = useState(1800); // 30 minutes
  const [assessmentStarted, setAssessmentStarted] = useState(false);
  const [assessmentSubmitted, setAssessmentSubmitted] = useState(false);
  const [selectedRole, setSelectedRole] = useState(""); 
  const [questions, setQuestions] = useState([]); 
  const [assessmentId, setAssessmentId] = useState(""); 
  const router = useRouter();
  const { toast } = useToast();

  useEffect(() => {
    const initialize = async () => {
      await initializeToken();
      setLoading(false);
    };
    initialize();
  }, [initializeToken]);

  useEffect(() => {
    if (!loading && !token) {
      router.push("/sign-up");
    }
  }, [loading, token, router]);

  useEffect(() => {
    if (user?.selectedRole) {
      setSelectedRole(user.selectedRole);
    }
  }, [user]);

  // --- CHANGE 2: Update how questions are retrieved ---
  useEffect(() => {
    if (selectedRole) {
      // Use the new allQuestions object with selectedRole as the key
      const roleQuestions = allQuestions[selectedRole] || [];
      setQuestions(roleQuestions);
      setUserAnswers(Array(roleQuestions.length).fill(null));
      setCurrentQuestionIndex(0);
    }
  }, [selectedRole]);

  useEffect(() => {
    let timer;
    if (assessmentStarted && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft((prevTime) => prevTime - 1);
      }, 1000);
    } else if (timeLeft === 0) {
      toast({
        title: "Time's up!",
        description: "Your assessment will be submitted now.",
        variant: "warning",
      });
      handleSubmitAssessment();
    }
    return () => clearInterval(timer);
  }, [assessmentStarted, timeLeft, toast]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? `0${remainingSeconds}` : remainingSeconds}`;
  };

  const handleStartAssessment = async () => {
    try {
      const startAssessmentResponse = await startAssessment(token, selectedRole);
      if (startAssessmentResponse?.status === "success" && startAssessmentResponse?.data._id) {
        setAssessmentId(startAssessmentResponse.data._id);
        toast({ title: "Assessment Started", description: "You have 30 minutes to complete the assessment.", variant: "default" });
        setAssessmentStarted(true);
        try { await document.documentElement.requestFullscreen(); } catch (error) { console.error("Fullscreen failed:", error); }
      } else if (startAssessmentResponse?.status === "failed") {
        toast({ title: "Assessment Already Completed", description: "You cannot start a new assessment.", variant: "warning" });
      } else {
        throw new Error(startAssessmentResponse?.message || "Failed to start assessment.");
      }
    } catch (error) {
 toast({ 
        title: "Error Starting Assessment", 
        description: error.message || "An unknown error occurpurple.", // Rely only on the 'error' object.
        variant: "destructive" 
      });    }
  };

  const handleSubmitAssessment = async () => {
    const questionsData = questions.map((question, index) => ({
      questionId: question.id, 
      answer: userAnswers[index],
      isCorrect: userAnswers[index] === question.correctAnswer,
    }));

    try {
      await submitAssessment(token, assessmentId, questionsData);
      setAssessmentSubmitted(true);
      if (document.fullscreenElement) { document.exitFullscreen(); }
      router.push("/complete");
    } catch (error) {
      toast({ title: "Error Submitting Assessment", description: error.message, variant: "error" });
    }
  };

  const handleAnswerChange = (event) => {
    const updatedAnswers = [...userAnswers];
    updatedAnswers[currentQuestionIndex] = event.target.value;
    setUserAnswers(updatedAnswers);
  };

  const handleNextQuestion = () => {
    if (userAnswers[currentQuestionIndex] !== null) {
      setCurrentQuestionIndex((prevIndex) => Math.min(prevIndex + 1, questions.length - 1));
    } else {
      toast({ title: "No Answer Selected", description: "Please select an option before proceeding to the next question.", variant: "warning" });
    }
  };

  const handlePreviousQuestion = () => {
    setCurrentQuestionIndex((prevIndex) => Math.max(prevIndex - 1, 0));
  };

  const handleLogout = () => {
    logout();
    router.push("/sign-in");
  };

  useEffect(() => {
    const handleTabChange = () => {
      if (assessmentStarted) {
        trackTabChange(token, user?.id, assessmentId); 
        toast({ title: "Tab Change Detected", description: "You have left the assessment window!", variant: "warning" });
      }
    };

    window.addEventListener("blur", handleTabChange);
    return () => {
      window.removeEventListener("blur", handleTabChange);
    };
  }, [assessmentStarted, assessmentId, token, user, toast]);

  if (loading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="animate-spin w-8 h-8" />
      </div>
    );
  }

  const isLastQuestion = currentQuestionIndex === questions.length - 1;
  const isAnswerSelected = userAnswers[currentQuestionIndex] !== null;
  const allQuestionsAnswepurple = userAnswers.every((answer) => answer !== null);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      {!assessmentStarted ? (
        <div className="bg-white max-w-7xl p-6 my-8 rounded-lg text-start">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-2xl font-bold">Terms and Conditions</h1>
<Link
                    className={`panel text-none text-4xl mb-8 text-red-500 ${federo.className}`}
                    href={`/`}
                    style={{ width: 200 }}
                  >
                   <Logo/>
                  </Link>                  </div>
          <p className="mb-4 text-sm text-muted-foreground">You are logged in as {user?.email}</p>
          <p className="mb-4 text-sm font-semibold">Role: {selectedRole}</p>
          <ul className="list-disc mx-4">{assessmentRules?.map((rule, index) => (<li key={index} className="mb-2">{rule}</li>))}</ul>
          <div className="flex items-center  mb-4">
            <Checkbox
              checked={isTermsCheckboxChecked}
              onCheckedChange={(checked) => setIsTermsCheckboxChecked(checked)}
              className="mr-2"
            />
            <p className="mb-0 font-semibold">By participating in this online exam, you agree to abide by the instructions and rules mentioned above. Failure to comply, including the creation of multiple accounts or other suspicious activities, may result in your disqualification from the assessment process.
            </p>
          </div>
          <div className="flex justify-between items-center gap-4">
            <Button variant="outline" className="border-primary text-primary" onClick={handleLogout}>
              Logout
            </Button>

            <Button
              onClick={() => {
                setIsTermsAccepted(true);
                handleStartAssessment();
              }}
              disabled={!isTermsCheckboxChecked || !selectedRole}
            >
              Start Assessment
            </Button>
            
          </div>
        </div>
      ) : (
        <div className="w-full h-screen px-8 pt-4">
          <div className="flex justify-between items-center mb-4 w-full">
                  <Link
                    className={`panel text-none text-4xl mb-8 text-red-500 ${federo.className}`}
                    href={`/`}
                    style={{ width: 200 }}
                  >
                   <Logo/>
                  </Link>                    <span className="text-xl font-semibold text-red-500">{formatTime(timeLeft)}</span>
          </div>
          <div className="bg-white mt-28 rounded-lg w-full md:max-w-2xl p-6 mx-auto select-none">
            <div className="flex justify-between mb-4">
              <h2 className="text-xl font-semibold">Question {currentQuestionIndex + 1} of {questions.length}</h2>
            </div>
            <div>
              <h3 className="text-xl mb-4">{questions[currentQuestionIndex]?.question}</h3>
              {questions[currentQuestionIndex]?.options?.map((option, index) => (
                <label key={index} className="flex items-center gap-2 cursor-pointer py-2 px-3 rounded-md hover:bg-gray-100">
                  <input
                    type="radio"
                    value={option}
                    checked={userAnswers[currentQuestionIndex] === option}
                    onChange={handleAnswerChange}
                    className="hidden" 
                  />
                  <div className={`w-full p-2 rounded-md ${userAnswers[currentQuestionIndex] === option ? "bg-red-200" : ""}`}>
                    {option}
                  </div>
                </label>
              ))}
            </div>
            <div className="flex justify-between items-center gap-4 mt-4">
              <Button
                variant="outline"
                onClick={handlePreviousQuestion}
                disabled={currentQuestionIndex === 0}
                className="border-primary text-primary hover:bg-none"
              >
                Previous
              </Button>
              {!isLastQuestion ? (
                <Button onClick={handleNextQuestion} disabled={!isAnswerSelected}>
                  Next
                </Button>
              ) : (
                <Button
                  onClick={handleSubmitAssessment}
                  disabled={!allQuestionsAnswepurple || assessmentSubmitted}
                  className="bg-green-500 text-white hover:bg-green-600"
                >
                  Submit Assessment
                </Button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Assessment;