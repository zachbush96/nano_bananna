# The following code should be pasted in a browser console. After 1 - 2 seconds, a zip file should be downloaded containing all of the images from the zillow listing.
## NOTE:
### 1. We first need to click the "See all photos" button to load the "media wall"
### 2. Once the media wall is loaded, we must scroll down all the way to the bottom to ensure all images are loaded
### Example of the "See all X photos" button that needs to be clicked first:
```
<button data-testid="gallery-see-all-photos-button" data-mst-cy="gallery-see-all-photos-button" class="StyledGallerySeeAllPhotosButton-fshdp-8-111-1__sc-167rdz3-0 exiHFe"><svg viewBox="0 0 32 32" aria-hidden="true" class="Icon-c11n-8-111-1__sc-13llmml-0 bXkBjv StyledGallerySeeAllPhotosButton__StyledGallerySeeAllPhotosIcon-fshdp-8-111-1__sc-167rdz3-1 csxIug" focusable="false" role="img"><path stroke="none" d="M12 4v8H4V4h8m1-2H3a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1V3a1 1 0 00-1-1zM28 4v8h-8V4h8m1-2H19a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1V3a1 1 0 00-1-1zM12 20v8H4v-8h8m1-2H3a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1V19a1 1 0 00-1-1zM28 20v8h-8v-8h8m1-2H19a1 1 0 00-1 1v10a1 1 0 001 1h10a1 1 0 001-1V19a1 1 0 00-1-1z"></path></svg> See all 45 photos</button>
```

### Javascript code to download images from a Zillow listing's media wall
```
const TARGET_FORMAT = "jpeg";  // Options: `jpeg` or `webp`
const TARGET_SIZE = "1536";  // Options: `1536`, `1344`, `1152`, `960`, `768`, `576`, `384`, `192`

// Load JSZip library
const script = document.createElement('script');
script.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.7.1/jszip.min.js";
document.head.appendChild(script);

script.onload = function() {
    // Function to download the zip file
    function downloadZip(zip) {
        zip.generateAsync({type: 'blob'}).then(function(content) {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(content);
            link.download = 'images.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    // Function to gather and zip image URLs from "media wall"
    function gatherAndZipImages() {
        // Gather the image URLs
        const mediaWall = document.querySelector('div[data-testid="hollywood-vertical-media-wall"]');
        const sources = Array.from(mediaWall.querySelectorAll(`source[type="image/${TARGET_FORMAT}"]`));

        // Try to pull the largest src URL from a source's srcset
        // srcset is in the format "<url> <size>, <url> <size>" so we split it and try to grab the last (hopefully largest) URL
        // It shouldn't really matter, though, since the regex will replace the target size with the largest possible anyway
        const imageUrls = sources.map(source => {return source.srcset.split(",").at(-1).split(" ")[1].replaceAll(/_\d+.(jpg|webp)/g, `_${TARGET_SIZE}.${TARGET_FORMAT}`)});

        const zip = new JSZip();
        const imgFolder = zip.folder("images");

        if (imageUrls.length > 0) {
            console.log('Image URLs:', imageUrls);
            const downloadPromises = imageUrls.map((url, index) => {
                return fetch(url).then(response => response.blob()).then(blob => {
                    imgFolder.file(`image_${index + 1}.${TARGET_FORMAT}`, blob);
                });
            });

            Promise.all(downloadPromises).then(() => {
                downloadZip(zip);
            });
        } else {
            console.log(`No .${TARGET_FORMAT} images found.`);
        }
    }

    // Execute the function to gather and zip images
    gatherAndZipImages();
}
```

