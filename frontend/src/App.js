import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { AppProvider } from "./AppContext";  // Import the AppProvider
import Home from "./Home";
import FriendsPage from "./FriendsPage";

function App() {
  return (
    <AppProvider> {/* Wrap the Router with AppProvider */}
      <Router>
        <Routes>
          <Route path="/home" element={<Home />} />
          <Route path="/friends" element={<FriendsPage />} />
        </Routes>
      </Router>
    </AppProvider>
  );
}

export default App;