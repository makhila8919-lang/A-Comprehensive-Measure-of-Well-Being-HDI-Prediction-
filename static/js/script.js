document.addEventListener("DOMContentLoaded", function () {

    console.log("HDI Prediction System Loaded");

    const form = document.querySelector("form");

    if (form) {

        form.addEventListener("submit", function () {

            const button = document.querySelector("button[type='submit']");

            button.innerHTML = "Predicting...";

            button.disabled = true;

        });

    }

});
