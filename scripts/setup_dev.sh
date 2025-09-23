#!/bin/bash

# CRM Development Environment Setup Script

echo "🚀 Setting up AI-Assisted CRM Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3.11+ is available
echo -e "${BLUE}📋 Checking Python version...${NC}"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VERSION" < "3.11" ]]; then
        echo -e "${RED}❌ Python 3.11+ required. Current version: $PYTHON_VERSION${NC}"
        exit 1
    fi
    PYTHON_CMD="python3"
else
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version OK${NC}"

# Check if we're in the right directory
if [[ ! -f "CLAUDE.md" ]]; then
    echo -e "${RED}❌ Please run this script from the CRM project root directory${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
cd backend
if [[ ! -d "venv" ]]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if Docker is running
echo -e "${BLUE}🐳 Checking Docker services...${NC}"
if docker ps > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running${NC}"

    # Start database services if not running
    cd ../docker
    if ! docker-compose ps | grep -q "Up"; then
        echo -e "${BLUE}🚀 Starting database services...${NC}"
        docker-compose up -d postgres redis
        echo -e "${GREEN}✅ Database services started${NC}"

        # Wait for services to be ready
        echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"
        sleep 10
    else
        echo -e "${GREEN}✅ Database services already running${NC}"
    fi
    cd ../backend
else
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Run setup tests
echo -e "${BLUE}🧪 Running setup tests...${NC}"
$PYTHON_CMD test_setup.py

if [[ $? -eq 0 ]]; then
    echo -e "\n${GREEN}🎉 Development environment setup complete!${NC}"
    echo -e "\n${BLUE}🚀 To start developing:${NC}"
    echo -e "   ${YELLOW}cd backend${NC}"
    echo -e "   ${YELLOW}source venv/bin/activate${NC}"
    echo -e "   ${YELLOW}uvicorn app.main:app --reload${NC}"
    echo -e "\n${BLUE}📚 Useful URLs:${NC}"
    echo -e "   Dashboard: ${YELLOW}http://localhost:8000${NC}"
    echo -e "   API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "   Health:    ${YELLOW}http://localhost:8000/health${NC}"
else
    echo -e "\n${RED}❌ Setup tests failed. Please check the logs above.${NC}"
    exit 1
fi