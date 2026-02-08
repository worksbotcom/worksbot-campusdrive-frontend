"use client";
import { useEffect, useState } from "react";
import { fetchAllUsers } from "@/utils/api";
import { useRouter } from "next/navigation";
import { CSVLink } from "react-csv";
import { Button } from "../ui/button";
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

// CSV Headers
const csvHeaders = [
  { label: "Name", key: "name" },
  { label: "Email", key: "email" },
  { label: "Phone", key: "phone" },
  { label: "College Name", key: "collegeName" },
  { label: "Degree", key: "degree" },
  { label: "Assessment Role", key: "assessmentRole" },
  { label: "Score", key: "score" },
  { label: "Tab Switch Count", key: "tabSwitchCount" },
  { label: "Created At", key: "createdAt" },
];

const formatDate = (dateString) => {
  const date = new Date(dateString); // Convert to JS Date object
  return date.toLocaleString('en-GB', { timeZone: 'Asia/Kolkata', hour12: true });
};


const SearchUsersPage = () => {
  const { token, initializeToken, logout } = useStore();
  const [users, setUsers] = useState([]);
  const [filteredUsers, setFilteredUsers] = useState([]);
  const [error, setError] = useState("");
  const [collegeFilter, setCollegeFilter] = useState("");
  const [roleFilter, setRoleFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();

  // Initialize token on mount
  useEffect(() => {
    initializeToken();
  }, []);

  // Fetch all users when token is available
  useEffect(() => {
    const getUsers = async () => {
      try {
        const data = await fetchAllUsers(token);
        setUsers(data.data);
        setFilteredUsers(data.data); // Initially set to all users
      } catch (err) {
        setError(err.message || "Failed to fetch users");
      }
    };
    if (token) getUsers();
  }, [token]);

  // Redirect to login if no token
  if (!token) {
    router.push("/sign-in");
    return null;
  }
  // Handle filtering and searching
  const handleFilter = () => {
    let filtered = users;

    if (collegeFilter) {
      filtered = filtered.filter((user) => user.collegeName === collegeFilter);
    }

    if (roleFilter) {
      filtered = filtered.filter((user) =>
        user.assessments.some((a) => a.role === roleFilter)
      );
    }

    if (searchQuery) {
      filtered = filtered.filter(
        (user) =>
          user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          user.email.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    setFilteredUsers(filtered);
  };

  // Prepare CSV data (flatten assessment data)
  const prepareCSVData = () =>
    filteredUsers.map((user) => {
      const assessment = user.assessments?.[0] || {}; // Use first assessment if available
      return {
        name: user.name,
        email: user.email,
        phone: user.phone,
        collegeName: user.collegeName,
        degree: user.degree,
        assessmentRole: assessment.role || "N/A",
        score: assessment.score || "N/A",
        tabSwitchCount: assessment.tabChanges?.[0]?.tabSwitchCount || "N/A",
        createdAt: assessment.createdAt ? formatDate(assessment.createdAt) : "N/A",
      };
    });

  return (
    <div>
     <Header/>
      <h1 className="text-center text-2xl font-bold mt-5">Search Users</h1>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {/* Filters and Search */}
      <div className="flex gap-4 mt-5 justify-center">
        <input
          type="text"
          placeholder="Search by Name or Email"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="border rounded p-2"
        />
        <input
          type="text"
          placeholder="Filter by College"
          value={collegeFilter}
          onChange={(e) => setCollegeFilter(e.target.value)}
          className="border rounded p-2"
        />
        <input
          type="text"
          placeholder="Filter by Assessment Role"
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="border rounded p-2"
        />
        <Button variant="secondary" onClick={handleFilter}>
          Apply Filter
        </Button>

        <Button>
          <CSVLink
            data={prepareCSVData()}
            headers={csvHeaders}
            filename="filtered_users.csv"
          >
            Download CSV
          </CSVLink>
        </Button>
      </div>

      {/* Users Table */}
      <div className="max-w-7xl mx-auto mt-10">
     
      <div>
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>Name</TableHead>
        <TableHead>Email</TableHead>
        <TableHead>Phone</TableHead>
        <TableHead>College</TableHead>
        <TableHead>Degree</TableHead>
        <TableHead>Assessment Role</TableHead>
        <TableHead>Score</TableHead>
        <TableHead>Tab Switch Count</TableHead>
        <TableHead>Created At</TableHead>
      </TableRow>
    </TableHeader>

    <TableBody>
      {filteredUsers.map((user) =>
        user.assessments && user.assessments.length > 0 ? (
          user.assessments.map((assessment, index) => (
            <TableRow key={`${user._id}-assessment-${index}`}>
              <TableCell>{user.name}</TableCell>
              <TableCell>{user.email}</TableCell>
              <TableCell>{user.phone}</TableCell>
              <TableCell>{user.collegeName}</TableCell>
              <TableCell>{user.degree}</TableCell>
              <TableCell>{assessment.role || "N/A"}</TableCell>
              <TableCell>{assessment.score || "N/A"}</TableCell>
              <TableCell>
                {assessment.tabChanges?.[0]?.tabSwitchCount || "N/A"}
              </TableCell>
              <TableCell>{assessment.createdAt ? formatDate(assessment.createdAt) : "N/A"}</TableCell>
              </TableRow>
          ))
        ) : (
          // If no assessments, show one row with user data and placeholders
          <TableRow key={`${user._id}-no-assessment`}>
            <TableCell>{user.name}</TableCell>
            <TableCell>{user.email}</TableCell>
            <TableCell>{user.phone}</TableCell>
            <TableCell>{user.collegeName}</TableCell>
            <TableCell>{user.degree}</TableCell>
            <TableCell colSpan={4} className="text-center">
              No assessments available
            </TableCell>
          </TableRow>
        )
      )}
    </TableBody>
  </Table>
</div>


      </div>
    </div>
  );
};

export default SearchUsersPage;
