import React from "react";
import "../styles/FriendsPage.css";

const VotersToggle = ({ currentVoter, setCurrentVoter }) => {
    const voters1 = [
        { id: 0, name: "אימא", icon: "🐶" },
        { id: 1, name: "אבא", icon: "🐤" },
        { id: 2, name: "שירה", icon: "🐼" },
        { id: 3, name: "רועי", icon: "🦁" },
    ];

    const voters2 = [
        { id: 0, name: "אליה", icon: "🐡" },
        { id: 1, name: "אבא", icon: "🐔" },
        { id: 2, name: "אימא", icon: "🦄" },
        { id: 3, name: "שיר", icon: "🐨" },
    ];

    return (
        <div className="voters-toggle">
            {voters2.map((voter) => (
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

