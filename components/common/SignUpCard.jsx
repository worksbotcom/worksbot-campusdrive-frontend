"use client";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { generateOTP, registerUser, verifyOTP } from "@/utils/api"; // Import verifyOTP
import useStore from "@/store/store";
import { useToast } from "@/hooks/use-toast";
import { useRouter } from "next/navigation";
import PhoneInput from "react-phone-input-2";
import "react-phone-input-2/lib/style.css";
import { degreeDepartments } from "@/utils/data";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  InputOTP,
  InputOTPGroup,
  InputOTPSlot,
} from "@/components/ui/input-otp";
import { questions as availableRoles } from "@/utils/questions";

const SignUpCard = () => {
  const [step, setStep] = useState(1); // Step tracking
  const [formData, setFormData] = useState({
    name: "",
    phone: "",
    alternativePhone: "",
    email: "",
    password: "",
    passOutYear: "",
    gender: "",
    collegeName: "",
    degree: "",
    department: "",
    location: "",
    selectedRole: "",
  });
  const [resumeFile, setResumeFile] = useState(null); // ADDED: State for resume file

  const [otp, setOtp] = useState(""); // State for OTP
  const [emailValidation, setEmailValidation] = useState({
    isValid: true,
    message: "",
  });
  const [phoneValidation, setPhoneValidation] = useState({
    isValid: true,
    message: "",
  });
  const [isEmailChecking, setIsEmailChecking] = useState(false);
  const [loading, setLoading] = useState(false);
  const { setEmail, setToken } = useStore();
  const { toast } = useToast();
  const router = useRouter();

  // Helper function to check if all required fields are filled
  const isFormStep1Valid = () => {
    const { name, phone, email, password, location } = formData;
    if (!name || !phone || !email || !password || !location) {
      toast({
        title: "Incomplete Information",
        description: "Please fill out all required fields.",
      });
      return false;
    }
    return true;
  };

  const roleNames = Object.keys(availableRoles);

  const isFormStep2Valid = () => {
    const { passOutYear, gender, collegeName, degree, department, selectedRole } = formData;
    if (!passOutYear || !gender || !collegeName || !degree || !department || !selectedRole) {
      toast({
        title: "Incomplete Information",
        description: "Please fill out all required fields.",
      });
      return false;
    }
    if (!resumeFile) {
        toast({
            title: "Resume Required",
            description: "Please upload your resume to continue."
        });
        return false;
    }
    return true;
  };

  const isOtpValid = () => {
    if (!otp) {
      toast({
        title: "Invalid OTP",
        description: "Please enter the OTP sent to your email.",
      });
      return false;
    }
    return true;
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };
const handleFileChange = (e) => {
    setResumeFile(e.target.files[0]);
  };

  const handleAlternativePhoneChange = (value) => {
    setFormData({ ...formData, alternativePhone: value });
  };
  
  const handlePhoneChange = (value) => {
    setFormData({ ...formData, phone: value });
    setPhoneValidation({ isValid: true, message: "" });
  };

  const handleSelectChange = (name, value) => {
    setFormData({ ...formData, [name]: value });
  };

  const handleNextStep = (e) => {
    e.preventDefault();
    if (isFormStep1Valid()) {
      setStep(2); // Move to step 2 if Step 1 is valid
    }
  };

  const handlePreviousStep = () => {
    setStep(1); // Go back to step 1
  };

  const handleSignUp = async (e) => {
    e.preventDefault();

    if (isFormStep2Valid()) {
      setLoading(true);
      const data = new FormData();
      Object.keys(formData).forEach(key => {
        data.append(key, formData[key]);
      });
      data.append('resume', resumeFile); 

      try {
        const out = await registerUser(data);
        if (out.status === "success") {
          toast({
            title: "Success",
            description: "Account created successfully!",
          });

          router.push("/sign-in"); // Redirect to homepage or any other page
          // await generateOTP(email); // Send OTP
          // setStep(3); // Move to OTP verification step
        } else {
          toast({
            title: "Failed",
            description: "User already exists.",
          });
        }
        setLoading(false);
      } catch (error) {
        console.log(error);
        toast({
          title: "Failed",
          description: error.error,
        });
      }
      setLoading(false);
    }
  };

  const handleOTPVerification = async (e) => {
    e.preventDefault();
    if (isOtpValid()) {
      setLoading(true);
      try {
        const response = await verifyOTP(formData.email, otp); // Verify OTP
        if (response?.status === "success") {
          toast({
            title: "Success",
            description: "OTP verified successfully!",
          });

          // Store the token in sessionStorage and set it in the global store
          const token = response.data.token; // Assuming the token is returned in response
          sessionStorage.setItem("token", token); // Store token in session storage
          setToken(token); // Set the token in global state

          router.push("/"); // Redirect to homepage or any other page
        } else {
          toast({
            title: "Failed",
            description: "Invalid OTP. Please try again.",
          });
        }
      } catch (error) {
        toast({
          title: "Try again!",
          description: error.error,
        });
      }
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto grid w-[350px] gap-4 overflow-y-auto">
      <div className="grid gap-2 text-center">
        <h1 className="text-3xl font-bold">Sign up</h1>
        <p className="text-muted-foreground">
          Create your account by filling out the form below.
        </p>
      </div>

      {/* Stepper */}
      {step === 1 && (
        <form onSubmit={handleNextStep} className="grid mx-1 gap-4">
          {/* User Information Step */}
          <div className="form-group">
            <Label htmlFor="name">Full Name</Label>
            <Input
              type="text"
              id="name"
              name="name"
              placeholder="Enter your fullname"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <Label htmlFor="phone">Phone Number</Label>
            <PhoneInput
              country={"in"}
              value={formData.phone}
              onChange={handlePhoneChange}
            />
            {!phoneValidation.isValid && (
              <span className="text-danger">{phoneValidation.message}</span>
            )}
          </div>
 <div className="form-group">
            <Label htmlFor="alternativePhone">Alternative Phone (Optional)</Label>
            <PhoneInput
              country={"in"}
              value={formData.alternativePhone}
              onChange={handleAlternativePhoneChange}
            />
          </div>

          <div className="form-group">
            <Label htmlFor="email">Email</Label>
            <Input
              type="email"
              id="email"
              name="email"
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
              required
            />
            {isEmailChecking && <span>Checking email...</span>}
            {!emailValidation.isValid && !isEmailChecking && (
              <span className="text-danger">{emailValidation.message}</span>
            )}
          </div>

          <div className="form-group">
            <Label htmlFor="password">Password</Label>
            <Input
              type="password"
              id="password"
              name="password"
              placeholder="Enter your password"
              value={formData.password}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <Label htmlFor="location">Location</Label>
            <Input
              type="text"
              id="location"
              name="location"
              placeholder="Your location"
              value={formData.location}
              onChange={handleChange}
              required
            />
          </div>

          <Button type="submit" className="w-full">
            Next
          </Button>
        </form>
      )}

      {step === 2 && (
        <form onSubmit={handleSignUp} className="grid mx-1 gap-2">
          {/* Education Details Step */}
          <div className="row form-group">
            <div className="form-group col-md-6">
              <Label htmlFor="passOutYear">Year of Passed Out</Label>
              <Select
                onValueChange={(value) =>
                  handleSelectChange("passOutYear", value)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Year of Passed Out" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2030">2030</SelectItem>
                  <SelectItem value="2029">2029</SelectItem>
                  <SelectItem value="2028">2028</SelectItem>
                  <SelectItem value="2027">2027</SelectItem>
                  <SelectItem value="2026">2026</SelectItem>
                  <SelectItem value="2025">2025</SelectItem>
                  <SelectItem value="2024">2024</SelectItem>
                  <SelectItem value="2023">2023</SelectItem>
                  <SelectItem value="2022">2022</SelectItem>
                  <SelectItem value="2021">2021</SelectItem>
                  <SelectItem value="2020">2020</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="form-group col-md-6">
              <Label htmlFor="gender">Gender</Label>
              <Select
                onValueChange={(value) => handleSelectChange("gender", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Gender" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Male">Male</SelectItem>
                  <SelectItem value="Female">Female</SelectItem>
                  <SelectItem value="Other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

           <div className="form-group">
            <Label htmlFor="collegeName">College Name</Label>
            <Select
              onValueChange={(value) => handleSelectChange("collegeName", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select College Name" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Annai Vailankanni College of Engineering">
                  Annai Vailankanni College of Engineering
                </SelectItem>
                <SelectItem value="Rathinam College of Arts and Science">
                  Rathinam College of Arts and Science
                </SelectItem>
                <SelectItem value="Rathinam Technical Campus">
                  Rathinam Technical Campus
                </SelectItem>
                <SelectItem value="Sri Shanmugha College of Engineering and Technology">
                  Sri Shanmugha College of Engineering and Technology
                </SelectItem>
                <SelectItem value="Nandha Engineering college">
                  Nandha Engineering college
                </SelectItem>
                 <SelectItem value="Nandha college of technology">
                  Nandha college of technology
                </SelectItem>
                <SelectItem value="Bannari Amman Institute of Technology">
                  Bannari Amman Institute of Technology
                </SelectItem>
                <SelectItem value="Sri Venkateshwara institute of science and technology">
                  Sri Venkateshwara institute of science and technology
                </SelectItem>
                <SelectItem value="Thiruvalluvar College of Engineering and technology">
                  Thiruvalluvar College of Engineering and technology
                </SelectItem>
                <SelectItem value="Ramco Institute of technology">
                  Ramco Institute of technology
                </SelectItem>
                <SelectItem value="Er perumal manimekalai college of engineering">
                  Er perumal manimekalai college of engineering
                </SelectItem>
                <SelectItem value="Mother Teresa  Women's University">
                  Mother Teresa  Women's University
                </SelectItem>
                <SelectItem value="Jkk nataraja college of engineering and technology">
                  Jkk nataraja college of engineering and technology
                </SelectItem>
                <SelectItem value="Fathima Michael college of engineering and technology">
                 Fathima Michael college of engineering and technology
                </SelectItem>
                <SelectItem value="Vv college of engineering">
                 Vv college of engineering
                </SelectItem>
                <SelectItem value="Bannari amman institute of technology">
                 Bannari amman institute of technology
                </SelectItem>
                <SelectItem value="Adarsh college of engineering">
                 Adarsh college of engineering
                </SelectItem>
                <SelectItem value="Daita Mahdusudana Sastry Sri Venkateshwara Hindu College of engineering(DMS SVH)">
                 Daita Mahdusudana Sastry Sri Venkateshwara Hindu College of engineering(DMS SVH)
                </SelectItem>
                <SelectItem value="Sri padmavathi mahila visvavidyalayam">
                 Sri padmavathi mahila visvavidyalayam
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="form-group">
            <Label htmlFor="degree">Degree</Label>
            <Select
              onValueChange={(value) => handleSelectChange("degree", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Degree" />
              </SelectTrigger>
              <SelectContent>
                {degreeDepartments.map((item) => (
                  <SelectItem key={item.degree} value={item.degree}>
                    {item.degree}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="form-group">
            <Label htmlFor="department">Department</Label>
            <Select
              onValueChange={(value) => handleSelectChange("department", value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Department" />
              </SelectTrigger>
              <SelectContent>
                {degreeDepartments
                  .find((item) => item.degree === formData.degree)
                  ?.departments.map((department) => (
                    <SelectItem key={department} value={department}>
                      {department}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>
          <div className="form-group">
            <Label htmlFor="selectedRole">Role</Label>
            <Select
              onValueChange={(value) =>
                handleSelectChange("selectedRole", value)
              }
            >
              <SelectTrigger>
                <SelectValue placeholder="Select Role" />
              </SelectTrigger>
               <SelectContent>
    {roleNames.map((role) => (
        <SelectItem key={role} value={role}>
            {role}
        </SelectItem>
    ))}
</SelectContent>
            </Select>
          </div>

          <div className="form-group">
            <Label htmlFor="resume">Upload Resume</Label>
            <Input 
              type="file" 
              id="resume" 
              name="resume"
              onChange={handleFileChange}
              accept=".pdf,.doc,.docx"
              required 
            />
          </div>

          <div className="flex justify-between items-center gap-4">
            <Button
              variant="secondary"
              type="button"
              onClick={handlePreviousStep}
            >
              Previous
            </Button>
            <Button type="submit" className="">
              {loading ? (
                <Loader2 className="animate-spin" />
              ) : (
                "Create account"
              )}
            </Button>
          </div>
        </form>
      )}

      {/* OTP Verification Step */}
      {step === 3 && (
        <form onSubmit={handleOTPVerification} className="grid gap-4">
          <div className="form-group">
            <Label htmlFor="otp">Enter OTP</Label>
            <div className="grid gap-2 justify-center">
              <InputOTP
                maxLength={6}
                value={otp}
                onChange={(e) => {
                  setOtp(e);
                }}
              >
                <InputOTPGroup>
                  {[...Array(6)].map((_, idx) => (
                    <InputOTPSlot key={idx} index={idx} />
                  ))}
                </InputOTPGroup>
              </InputOTP>
            </div>
          </div>
          <Button type="submit" className="w-full">
            {loading ? <Loader2 className="animate-spin" /> : "Create account"}
          </Button>
        </form>
      )}

      <div className="mt-2 text-center text-sm">
        Already have an account?{" "}
        <Link href="/sign-in" className="underline">
          Sign in
        </Link>
      </div>
    </div>
  );
};

export default SignUpCard;
