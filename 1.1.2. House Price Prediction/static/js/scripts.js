// static/js/scripts.js

document.addEventListener("DOMContentLoaded", function () {
  // Initialize homepage carousel with 2-second interval
  var carouselElement = document.querySelector("#homeCarousel");
  
  if (carouselElement && typeof bootstrap !== "undefined") {
    try {
      var carousel = bootstrap.Carousel.getOrCreateInstance(carouselElement, {
        interval: 2000,
        ride: "carousel"
      });
    } catch (error) {
      console.warn("Error initializing carousel:", error);
    }
  }
});
