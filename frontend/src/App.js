import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./Home";
import FriendsPage from "./FriendsPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/home" element={<Home />} />
        <Route path="/friends" element={<FriendsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
