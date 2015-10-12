'use strict';



/*******************************************
 Network canvas animation
 Adapted from:
 *******************************************/

(function() {

  let width, height, largeHeader, canvas, ctx, points, target, animateHeader = true;

  // Main
  initHeader();
  initAnimation();

  function initHeader() {
    width = window.innerWidth;
    height = 300;
    target = {x: width/2, y: height/2};

    canvas = document.getElementById('network-canvas');
    canvas.width = width;
    canvas.height = height;
    ctx = canvas.getContext('2d');

    // create points
    points = [];
    for (let x = 0; x < width; x = x + width/4) {
      for (let y = 0; y < height; y = y + height/5) {
        let px = x + Math.random()*width/10;
        let py = y + Math.random()*height/5;
        let p = { x: px, originX: px, y: py, originY: py };
        points.push(p);
      }
    }

    for (let i = 0; i < points.length; i++) {
      let closest = [];
      let p1 = points[i];
      for (let j = 0; j < points.length; j++) {
        let p2 = points[j];
        if (!(p1 == p2)) {
          let placed = false;
          for (let k = 0; k < 5; k++) {
            if (!placed) {
              if (closest[k] == undefined) {
                closest[k] = p2;
                placed = true;
              }
            }
          }

          for (let k = 0; k < 5; k++) {
            if (!placed) {
              if (getDistance(p1, p2) < getDistance(p1, closest[k])) {
                closest[k] = p2;
                placed = true;
              }
            }
          }
        }
      }
      p1.closest = closest;
    }

    for (let i in points) {
      let c = new Circle(points[i], 3+Math.random()*3, 'rgba(255,255,255,0.2)');
      points[i].circle = c;
    }
  }

  function initAnimation() {
    animate();
    for (let i in points) {
      shiftPoint(points[i]);
    }
  }

  function animate() {
    if (animateHeader) {
      ctx.clearRect(0,0,width,height);
      for (let i in points) {
        if (Math.abs(getDistance(target, points[i])) < 10000) {
          points[i].active = 0.4;
          points[i].circle.active = 0.8;
        } else if (Math.abs(getDistance(target, points[i])) < 100000) {
          points[i].active = 0.1;
          points[i].circle.active = 0.3;
        } else if (Math.abs(getDistance(target, points[i])) < 20000) {
          points[i].active = 0.02;
          points[i].circle.active = 0.1;
        } else {
          points[i].active = 0;
          points[i].circle.active = 0;
        }

        drawLines(points[i]);
        points[i].circle.draw();
      }
    }
    requestAnimationFrame(animate);
  }

  function shiftPoint(p) {
    TweenLite.to(p, 1+1*Math.random(), {
      x: p.originX-50+Math.random()*100,
      y: p.originY-50+Math.random()*100,
      ease: Circ.easeInOut,
      onComplete: function() {
        shiftPoint(p);
      }
    });
  }

  // Canvas manipulation
  function drawLines(p) {
    if (!p.active) return;
    for (let i in p.closest) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.closest[i].x, p.closest[i].y);
      ctx.strokeStyle = 'rgba(220,220,220,'+ p.active+')';
      ctx.stroke();
    }
  }

  function Circle(pos, rad, color) {
    let _this = this;

    (function() {
      _this.pos = pos || null;
      _this.radius = rad || null;
      _this.color = color || null;
    })();

    this.draw = function() {
      if (!_this.active) return;
      ctx.beginPath();
      ctx.arc(_this.pos.x, _this.pos.y, _this.radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = 'rgba(220,220,220,'+ _this.active+')';
      ctx.fill();
    };
  }

  function getDistance(p1, p2) {
    return Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2);
  }

})();
