# HeatSafe Oklahoma — GOES-19 GLM Lightning Ingestor

Production bridge between NOAA GOES-19 GLM Level 2 LCFA lightning files and the HeatSafe Oklahoma Base44 backend.

Data source: NOAA GOES-19 public S3 bucket `s3://noaa-goes19/`, product `GLM-L2-LCFA`.

Endpoints:
- `GET /health`
- `GET /lightning?bbox=minLat,minLon,maxLat,maxLon&since=<ISO-8601>&source=goes19`

`/lightning` requires `X-Api-Key: <LIGHTNING_INGESTOR_KEY>`.

Required production environment variable:
- `LIGHTNING_INGESTOR_KEY`

Railway should use the included Dockerfile, `/health` as the healthcheck, and run continuously.

HeatSafe Base44 must be configured with the Railway HTTPS URL and the same shared secret. Safety rule: **unknown is not clear**.

Deployment source is connected to Railway production.