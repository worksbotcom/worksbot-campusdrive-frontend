"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { CSVLink } from "react-csv";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import useStore from "@/store/store";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Header from "@/app/admin/_component/Header";
import { Loader2 } from "lucide-react";
import { fetchPaginatedResults, downloadResumes, downloadFullCSV } from "@/utils/api"; 

// Headers for the CSV export
const csvHeaders = [
  { label: "Name", key: "name" },
  { label: "Email", key: "email" },
  { label: "Phone", key: "phone" },
  { label: "College Name", key: "collegeName" },
  { label: "Degree", key: "degree" },
  { label: "Department", key: "department" },
  { label: "Assessment Role", key: "assessmentRole" },
  { label: "Score", key: "score" },
  { label: "Tab Switch Count", key: "tabSwitchCount" },
  { label: "Assessment Date", key: "createdAt" },
];

const AllUsersPage = () => {
  const { token, initializeToken } = useStore();
  const router = useRouter();

  // Component State
  const [loading, setLoading] = useState(true);
  const [isDownloadingResumes, setIsDownloadingResumes] = useState(false);
  const [isDownloadingCSV, setIsDownloadingCSV] = useState(false); // New state for CSV download
  const [paginatedUsers, setPaginatedUsers] = useState([]);

  // Filter State
   const [collegeName, setCollegeName] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");


  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalResults, setTotalResults] = useState(0);

  // Initialize token from store/storage
  useEffect(() => {
    initializeToken();
  }, [initializeToken]);

  // Fetch data when component mounts or filters/page changes
  useEffect(() => {
    if (token) {
      fetchData();
    }
  }, [token, currentPage, collegeName, startDate, endDate]);


  const fetchData = async () => {
    if (!token) return;
    setLoading(true);
    try {
      // UPDATED: Pass date range to the API
      const params = {
        page: currentPage,
        limit: 15,
        collegeName: collegeName || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
      };
      const data = await fetchPaginatedResults(token, params);
      setPaginatedUsers(data.data.users);
      setTotalPages(data.data.totalPages);
      setTotalResults(data.data.totalResults);
    } catch (err) {
      console.error("Failed to fetch results:", err.message);
      if(err?.response?.status === 401) {
          router.push('/sign-in');
      }
    } finally {
      setLoading(false);
    }
  };
  
  // Handlers
  const handleFilterSubmit = (e) => {
    e.preventDefault();
    setCurrentPage(1); // Reset to first page on new filter
    // The useEffect hook will automatically refetch data because filter states (collegeName/date) change
    fetchData(); // We can also call it directly to be more explicit
  };

  const handleDownloadResumes = async () => {
    if (!token || totalResults === 0) return;
    setIsDownloadingResumes(true);
    try {
      const params = {
        collegeName: collegeName || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
      };
      await downloadResumes(token, params);
    } catch (error) {
      console.error("Failed to download resumes:", error);
      alert("An error occurred while downloading resumes.");
    } finally {
      setIsDownloadingResumes(false);
    }
  };

  const handleDownloadCSV = async () => {
    if (!token || totalResults === 0) return;
    setIsDownloadingCSV(true);
    try {
      const params = {
        collegeName: collegeName || undefined,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
      };
      await downloadFullCSV(token, params);
    } catch (error) {
      console.error('Error downloading CSV:', error.response || error);
      alert("An error occurred while downloading the report.");
    } finally {
      setIsDownloadingCSV(false);
    }
  };

  // Helper Functions
  const convertUTCToIST = (utcDate) => {
    if (!utcDate) return "N/A";
    const date = new Date(utcDate);
    return date.toLocaleDateString("en-IN", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  if (!token) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Loader2 className="animate-spin w-8 h-8" />
      </div>
    );
  }

  return (
    <div>
      <Header />
      <div className="max-w-7xl mx-auto mt-10 p-4">
        <h1 className="text-2xl font-bold mb-4">Assessment Results Dashboard</h1>
        
        {/* Filter Section */}
        <form onSubmit={handleFilterSubmit} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end p-4 border rounded-lg mb-6 bg-gray-50">
          <div>
            <Label htmlFor="collegeName">Filter by College</Label>
            <Input id="collegeName" value={collegeName} onChange={(e) => setCollegeName(e.target.value)} placeholder="e.g., Satyam College" />
          </div>
          <div>
            <Label htmlFor="startDate">Start Date</Label>
            <Input id="startDate" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          </div>
          <div>
            <Label htmlFor="endDate">End Date</Label>
            <Input id="endDate" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          </div>
          <Button type="submit" className="w-full md:w-auto">Apply Filters</Button>
        </form>


        {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 mb-4">
          <Button onClick={handleDownloadCSV} disabled={isDownloadingCSV || totalResults === 0 || loading}>
            {isDownloadingCSV && <Loader2 className="animate-spin mr-2 h-4 w-4" />}
            Download Report (CSV)
          </Button>
          <Button onClick={handleDownloadResumes} disabled={isDownloadingResumes || totalResults === 0 || loading}>
            {isDownloadingResumes && <Loader2 className="animate-spin mr-2 h-4 w-4" />}
            Download Resumes ({totalResults})
          </Button>
        </div>

        {/* Results Table */}
        {loading ? (
          <div className="text-center p-10"><Loader2 className="animate-spin w-8 h-8 mx-auto text-primary" /></div>
        ) : (
          <>
            <div className="border rounded-lg overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>S.No.</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>College</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Score</TableHead>
                    <TableHead>Tabs</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {paginatedUsers.map((user, index) => {
                    const assessments = user.assessments || [];
                    let lowestScore = null, lowestScoreAssessment = null, highestTabSwitchCount = -1;
                    assessments.forEach(a => {
                        if (a.score > 0 && (lowestScore === null || a.score < lowestScore)) {
                            lowestScore = a.score;
                            lowestScoreAssessment = a;
                        }
                        const switches = a.tabChanges?.[0]?.tabSwitchCount || 0;
                        if(switches > highestTabSwitchCount) highestTabSwitchCount = switches;
                    });
                     if (lowestScore === null) {
                        lowestScore = "N/A";
                        lowestScoreAssessment = assessments.find(a => a.score === 0) || null;
                    }

                    return (
                      <TableRow key={user._id}>
                        <TableCell>{(currentPage - 1) * 15 + index + 1}</TableCell>
                        <TableCell>{user.name}</TableCell>
                        <TableCell>{user.email}</TableCell>
                        <TableCell>{user.collegeName}</TableCell>
                        <TableCell>{lowestScoreAssessment?.role || "N/A"}</TableCell>
                        <TableCell className="font-medium">{lowestScore}</TableCell>
                        <TableCell>{highestTabSwitchCount >= 0 ? highestTabSwitchCount : "N/A"}</TableCell>
                        <TableCell>{convertUTCToIST(lowestScoreAssessment?.createdAt)}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
            {paginatedUsers.length === 0 && <p className="text-center p-6 text-gray-500">No results found for the selected filters.</p>}

            {/* Pagination Controls */}
            {totalPages > 1 && (
                <div className="flex justify-between items-center mt-6">
                    <Button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage <= 1}>
                        Previous
                    </Button>
                    <span className="text-sm font-medium">
                        Page {currentPage} of {totalPages} ({totalResults} results)
                    </span>
                    <Button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage >= totalPages}>
                        Next
                    </Button>
                </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default AllUsersPage;