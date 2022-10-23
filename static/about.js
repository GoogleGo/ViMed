const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const html = document.documentElement;

canvas.width = window.innerWidth/3*2;
canvas.height = window.innerHeight/3*2;
canvas.style.marginLeft = `${window.innerWidth/6}px`;
let width = canvas.width;
let height = canvas.height;

const offSetUpper = 0.025;
const offSetLower = 1;

const frameCount = 298;
const currentFrame = index => (
  `static/Animations/${index.toString().padStart(6, '0')}.jpg`
)

const preloadImages = () => {
  for (let i = 1; i < frameCount; i++) {
    const img = new Image();
    img.src = currentFrame(i);
  }
};

const img = new Image()
img.src = currentFrame(1);
img.onload=function(){
  ctx.drawImage(img, 0, 0, width, height);
}

const updateImage = index => {
  img.src = currentFrame(index);
  ctx.drawImage(img, 0, 0, width, height);
}

window.addEventListener('scroll', () => {
  //el.style.backgroundPositionX = (document.height * offSetLower).toString() + "px"; //80% of the page height
  const scrollTop = html.scrollTop;
  const maxScrollTop = html.scrollHeight - window.innerHeight;
  const scrollFraction = scrollTop / maxScrollTop;

  var frame = 0; //scrollFraction * frameCount
  if (scrollFraction < offSetUpper) {
    frame = 0;
  } else if (scrollFraction > offSetLower) {
    frame = frameCount;
  } else {
    frame = ((scrollFraction - offSetUpper) * frameCount) * (1/offSetLower);
  }
  //If position of window isn't on the canvas, don't update the image
  if (scrollFraction < offSetUpper || scrollFraction > offSetLower) {
    canvas.style.position = 'absolute';
    return;
  } else {
    canvas.style.position = 'fixed';
  }

  frame = Math.ceil(frame);
  console.log(frame, scrollFraction < offSetUpper, scrollFraction > offSetLower, scrollFraction - offSetUpper * frameCount, scrollFraction);

  const frameIndex = Math.min(
    frameCount,
    frame
  );

  requestAnimationFrame(() => updateImage(frameIndex))
});

window.addEventListener('resize', () => {
  const scrollTop = html.scrollTop;
  const maxScrollTop = html.scrollHeight - window.innerHeight;
  const scrollFraction = scrollTop / maxScrollTop;
  canvas.width = window.innerWidth/3*2;
  canvas.height = window.innerHeight/3*2;
  canvas.style.marginLeft = `${window.innerWidth/6}px`;
  width = canvas.width;
  height = canvas.height;

  //Update image
  if (scrollFraction < offSetUpper) {
    requestAnimationFrame(() => updateImage(0))
  } else if (scrollFraction > offSetLower) {
    requestAnimationFrame(() => updateImage(frameCount))
  } else {

  window.scrollBy(0, 1);
  }
});

preloadImages()


