import React from "react";
import "../styles/FriendsPage.css";

const VotersToggle = ({ currentVoter, setCurrentVoter }) => {
    const voters = [
        { id: 0, name: "הילה", icon: "🐶" },
        { id: 1, name: "מיכלי", icon: "🐤" },
        { id: 2, name: "דנה היפה", icon: "🐼" },
        { id: 3, name: "ניצן", icon: "🦁" },
    ];

    return (
        <div className="voters-toggle">
            {voters.map((voter) => (
                <div className="tooltip-container" key={voter.id}>
                    <button
                        className={`voter-icon ${currentVoter === voter.id ? "active" : ""}`}
                        onClick={() => setCurrentVoter(voter.id)}
                    >
                        {voter.icon}
                    </button>
                    <span className="tooltip-text">{voter.name}</span>
                </div>
            ))}
        </div>
    );
};

export default VotersToggle;

