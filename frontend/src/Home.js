import React from "react";
import { useState, useEffect } from "react";
import Header from "./components/Header";
import ImageUploader from "./components/ImageUploader";
import OutputBox from "./components/OutputBox";
import UseSteps from "./components/UseSteps";
import "./styles/Home.css";


const descriptionText = `כל אחד צומח בדרך משלו – גלה את העץ שמספר את הסיפור שלך`;

const Home = () => {
    const [isHidden, setIsHidden] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            const contentSection = document.querySelector(".content");
            if (!contentSection) return;

            const contentTop = contentSection.getBoundingClientRect().top;
            setIsHidden(contentTop <= window.innerHeight * 0.8); // Hide when 80% of content is visible
        };

        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    const scrollToSection = () => {
        const targetSection = document.querySelector(".content");
        if (targetSection) {
            targetSection.scrollIntoView({ behavior: "smooth" });
        }
    };


    const [isStepsVisible, setIsStepsVisible] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            const stepsSection = document.querySelector(".steps-container");
            if (!stepsSection) return;

            const sectionTop = stepsSection.getBoundingClientRect().top;
            if (sectionTop < window.innerHeight * 0.8) {
                setIsStepsVisible(true);
            }
        };

      window.addEventListener("scroll", handleScroll);
      return () => window.removeEventListener("scroll", handleScroll);
    }, []);


  return (
    <div className="home-container">
      
      <Header />
      
      {/* <div className="shape shape1"></div>
      <div className="shape shape2"></div> */}
      
  
      <div className="image-header-container">
        <div className="image-header">
          <h2>
            יש לך שורשים חזקים או שאתה יותר ברוח החופשית?
          </h2>          
        </div>
      </div>
      

      <div className="image-description">
        <h3>
            {descriptionText}
        </h3>
      </div>

      {!isHidden && (
        <div className="start-button" onClick={scrollToSection}>
          <button>
            ?שנקפוץ למודל הקסם שלנו
          </button>
        </div>
      )}
      
      <UseSteps />

      <div className="content">
        <ImageUploader />
        <OutputBox />
      </div>

      

    <div className="bottom-banner"/>
    <footer className="footer">
        <p>© 2024 Tree Tails כל הזכויות שמורות</p>
      </footer>
    </div>
  );
};

export default Home;
