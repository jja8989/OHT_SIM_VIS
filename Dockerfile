# Use a specific Node.js version base image (21.2.0)
FROM node:21.2.0-alpine

# Set working directory
WORKDIR /app

# # Install dependencies
COPY ./package.json ./package-lock.json ./
RUN npm install --include=dev

# # Copy application code
COPY . .
# # Expose the Next.js server port
EXPOSE 3000

# # Run the Next.js app
CMD ["npm", "run", "dev"]
