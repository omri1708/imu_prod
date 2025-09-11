FROM nginx:alpine
COPY ./ui /usr/share/nginx/html
RUN addgroup -S ui && adduser -S ui -G ui && chown -R ui:ui /usr/share/nginx/html
USER ui
EXPOSE 80