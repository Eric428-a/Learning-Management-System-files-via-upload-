// static/js/scripts.js
document.addEventListener("DOMContentLoaded", function() {
    var myCarousel = document.querySelector('#carouselExample');
    if (myCarousel) {
      var carousel = new bootstrap.Carousel(myCarousel, {
        interval: 2000,
        ride: 'carousel'
      });
    }
  });
  