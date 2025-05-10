#include "OptixView.h"

//What should happen when the Window resizes
static void OnWindowResize(GLFWwindow* window, int width, int height) {
	OptixView* optixView = static_cast<OptixView*>(glfwGetWindowUserPointer(window));
	assert(optixView);

	glm::ivec2 size{ width, height };
	optixView->Resize(width, height);
}




OptixView::OptixView(OptixViewDefinition viewDef, glm::ivec2 viewSize) {
	this->window = std::unique_ptr<OpenGLWindow>(new OpenGLWindow(viewSize));
	this->optixRenderer = std::unique_ptr<OptixRenderer>(new OptixRenderer(viewDef.ptxPath));
	this->viewTexture = std::unique_ptr<OptixViewTexture>(new OptixViewTexture());

	this->viewSize = viewSize;
	this->Resize(viewSize);

	//Set this view as a custom Pointer for the window
	glfwSetWindowUserPointer(this->window->window, this);
	glfwSetFramebufferSizeCallback(this->window->window, OnWindowResize);
}

#pragma region Public

void OptixView::Run() {
	//Render Loop
	while (!glfwWindowShouldClose(this->window->window)) {
		//Optix
		this->DrawOptix();

		//OpenGL
		this->GLFrameSetup();
		this->DrawToWindow();
		this->EndFrame();
	}
}

void OptixView::Resize(int width, int height) {
	this->viewSize = glm::ivec2(width, height);
	this->optixRenderer->Resize(this->viewSize);
}

void OptixView::Resize(glm::ivec2 viewSize) {
	this->viewSize = viewSize;
	this->optixRenderer->Resize(this->viewSize);
}

#pragma endregion

#pragma region Private

void OptixView::GLFrameSetup() {
	//was in sample, but doesn't change anything
	//glDisable(GL_LIGHTING);
	//glColor3f(1, 1, 1);

	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	//Clear Screen
	glClearColor(1.0f, 1.0f, 1.0f, 1.00f);
	glClear(GL_COLOR_BUFFER_BIT);

	//needs no depthtest (displaying just an image)
	glDisable(GL_DEPTH_TEST);
}

void OptixView::DrawOptix() {
	std::vector<uint32_t> pixels;
	pixels.resize(this->viewSize.x * this->viewSize.y);

	//Render Optix and retrieve pixels
	this->optixRenderer->Render(pixels.data());
	this->viewTexture->SetData(pixels.data(), this->viewSize);
}

void OptixView::DrawToWindow() {

	//Set viweport size
	glViewport(0, 0, this->viewSize.x, this->viewSize.y);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, (float)this->viewSize.x, 0.f, (float)this->viewSize.y, -1.f, 1.f);

	this->viewTexture->BindTexture();

	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.f, 0.f);
		glVertex3f(0.f, 0.f, 0.f);

		glTexCoord2f(0.f, 1.f);
		glVertex3f(0.f, (float)this->viewSize.y, 0.f);

		glTexCoord2f(1.f, 1.f);
		glVertex3f((float)this->viewSize.x, (float)this->viewSize.y, 0.f);

		glTexCoord2f(1.f, 0.f);
		glVertex3f((float)this->viewSize.x, 0.f, 0.f);
	}
	glEnd();
}

void OptixView::EndFrame() {
	glfwSwapBuffers(this->window->window);
	glfwPollEvents();
}

#pragma endregion