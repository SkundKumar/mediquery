terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" # Ensure this matches your Bedrock region
}

# 1. IAM Role for Lambda to access Bedrock and CloudWatch
resource "aws_iam_role" "mediquery_v2_role" {
  name = "mediquery_v2_lambda_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

# 2. Grant permissions for Multimodal Bedrock and Telemetry Logs
resource "aws_iam_policy" "mediquery_v2_policy" {
  name        = "mediquery_v2_bedrock_cloudwatch_policy"
  description = "Allows Lambda to call Bedrock Nova Lite and log to CloudWatch"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:Converse"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mediquery_v2_attach" {
  role       = aws_iam_role.mediquery_v2_role.name
  policy_arn = aws_iam_policy.mediquery_v2_policy.arn
}

# 3. The Upgraded Serverless Compute (Lambda)
resource "aws_lambda_function" "mediquery_v2_engine" {
  function_name = "mediquery-engine-v2"
  role          = aws_iam_role.mediquery_v2_role.arn
  package_type  = "Image"
  image_uri     = "YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/mediquery:latest" # Update this later in CI/CD
  timeout       = 30 # Increased timeout for Vision Processing
  memory_size   = 1024 # Increased memory for image base64 decoding

  environment {
    variables = {
      PINECONE_API_KEY = "dummy-key-update-in-console"
      MY_AWS_REGION    = "us-east-1"
    }
  }
}

# 4. The Public Multimodal API Gateway
resource "aws_lambda_function_url" "mediquery_v2_url" {
  function_name      = aws_lambda_function.mediquery_v2_engine.function_name
  authorization_type = "NONE"

  cors {
    allow_credentials = true
    allow_origins     = ["*"]
    allow_methods     = ["POST"]
    allow_headers     = ["date", "keep-alive", "content-type"]
    max_age           = 86400
  }
}

# 5. Output the live URL to the terminal
output "mediquery_v2_api_url" {
  value = aws_lambda_function_url.mediquery_v2_url.function_url
}