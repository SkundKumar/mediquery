# 1. Use the official AWS Lambda Python 3.12 image
FROM public.ecr.aws/lambda/python:3.12

# 2. Copy only the requirements first (to use Docker's cache)
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# 3. Install dependencies
RUN pip install -r requirements.txt

# 4. Copy ONLY the application code (No .env, no data folders)
COPY query_mediquery.py ${LAMBDA_TASK_ROOT}

# 5. Set the CMD to your handler
CMD [ "query_mediquery.handler" ]