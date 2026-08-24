ARG NODE_BASE=node:22.22.1-bookworm-slim@sha256:4f77a690f2f8946ab16fe1e791a3ac0667ae1c3575c3e4d0d4589e9ed5bfaf3d
ARG NGINX_BASE=nginxinc/nginx-unprivileged:1.31.3-alpine3.24@sha256:f972e5322b9797dc2a6b830030094426437b1ae7032e4644496395336ac6fdac
FROM ${NODE_BASE} AS build

WORKDIR /web
RUN corepack enable
COPY facial_paralysis_web/package.json facial_paralysis_web/pnpm-lock.yaml facial_paralysis_web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY facial_paralysis_web/ ./
RUN pnpm test:run && pnpm build

FROM ${NGINX_BASE}
COPY facial_paralysis_web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist/ /usr/share/nginx/html/
USER 101:101
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/healthz"]
