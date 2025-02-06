import React, { useContext } from "react";
import "../styles/FriendsPage.css";

const VotingTable = ({ ratings, response }) => {
   
    // Function to generate star ratings
    const renderStars = (rating) => {
        const maxStars = 5;
        return (
            <span className="stars">
                {"★".repeat(rating)}{"☆".repeat(maxStars - rating)}
            </span>
        );
    };

    // Function to split the response into lines
    const splitResponseIntoLines = (responseText) => {
        return responseText.split("\n\n").map((line, index) => ({ id: index, line }));
    };

    const parsedResponse = splitResponseIntoLines(response);

    return (
        <div className="voting-table-container">
            <table>
                <thead>
                    <tr>
                        <th>?מה מודל הקסם שלנו קבע</th>
                        <th>?כמה צדקנו</th>
                    </tr>
                </thead>
                <tbody>
                    {/* Iterate over the parsed response and display each line */}
                    {parsedResponse.map((item) => (
                        <tr key={item.id}>
                            <td>{item.line}</td>
                            <td>{renderStars(ratings[item.id])}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default VotingTable;

