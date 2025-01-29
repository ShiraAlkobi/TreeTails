import React, { useState } from "react";
import Header from "./components/Header";
import ImageDisplay from "./components/ImageDisplay";
import VotersToggle from "./components/VotersToggle";
import VotingTable from "./components/VotingTable";
import "./styles/FriendsPage.css";

const FriendsPage = () => {
    const [currentVoter, setCurrentVoter] = useState(0);

    // Analysis points
    const analysisPoints = ["Confidence", "Stability", "Creativity"];
    const voterRatings = [
        [4, 5, 3], // Ratings by Voter 1
        [3, 4, 4], // Ratings by Voter 2
        [5, 5, 4], // Ratings by Voter 3
        [4, 4, 5], // Ratings by Voter 4
    ];

    return (
        <div className="friends-page-container">
            <Header />
            <div className="header-container">
                <div className="header-title">
                    <h2>
                        ברוכים הבאים לאזור החברים!
                    </h2>          
                </div>
            </div>

            <div className="header-sub-title">
                <h3>
                    כאן תוכלו לגלות ולשתף תוצאות עם חברים
                </h3>
            </div>

        
            <ImageDisplay />
            <VotersToggle currentVoter={currentVoter} setCurrentVoter={setCurrentVoter} />
            <VotingTable analysisPoints={analysisPoints} ratings={voterRatings[currentVoter]} />
            <footer className="footer">
                <p>© 2024 Tree Tails כל הזכויות שמורות</p>
            </footer>
        </div>
    );
};

export default FriendsPage;
