import React from "react";
import "../styles/Home.css";

const OutputBox = ({ response }) => {
  return (
    <div className="output-box">
      {response ? <p>{response}</p> : <p>מחכים לראות מי אתה</p>}
    </div>
  );
};

export default OutputBox;