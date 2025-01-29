import React from "react";
import { useNavigate } from "react-router-dom";
import logo from "../images/logo.png"; // make sure this path is correct
import friendsIcon from "../images/friends-zone.png"; // make sure this path is correct

const Header = () => {
  const navigate = useNavigate();

  return (
    <header className="header">
      <img src={logo} alt="Tree Tails Logo" className="logo" />
      <h1 className="title" onClick={() => navigate("/home")}
      >Tree Tails</h1>

      <div className="options-container">
        <div
        className="friends-zone-container"
        onClick={() => navigate("/friends")} // use navigate here
          >
            <h1 className="options-label">Friends Zone</h1>
        </div>

        <div className="options-divide">
          |
        </div>

        <div
        className="about-us-container"
        onClick={() => navigate("/friends")} // use navigate here
          >
            <h1 className="options-label">About Us</h1>
        </div>

        <div className="options-divide">
          |
        </div>
      </div>
      
    </header>
  );
};

export default Header;
