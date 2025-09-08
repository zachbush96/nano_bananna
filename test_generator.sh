curl -s http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Add more furniture to this game room/living area. Make it look orientated to a Pittsburgh sports fan.",
        "image_url":"https://photos.zillowstatic.com/fp/1d9de0ba3d98f6e3e71f8f689569fbee-cc_ft_1536.webp"}' \
  | jq

curl -s http://localhost:5000/zillow/download -H "Content-Type: application/json" 
-d '{"url":"https://www.zillow.com/homedetails/2030-Williams-Rd-Nolensville-TN-37135/42630211_zpid/", 
"format":"jpeg", "size":"1536"}' | jq