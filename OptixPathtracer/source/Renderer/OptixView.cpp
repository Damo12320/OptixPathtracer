#include "OptixView.h"
#include <iostream>

#pragma region GLFW_Callbacks
static void OnWindowResize(GLFWwindow* window, int width, int height) {
	OptixView* optixView = static_cast<OptixView*>(glfwGetWindowUserPointer(window));
	assert(optixView);

	glm::ivec2 size{ width, height };
	optixView->Resize(width, height);
}

static void OnCursorMove(GLFWwindow* window, double x, double y) {
	OptixView* optixView = static_cast<OptixView*>(glfwGetWindowUserPointer(window));
	assert(optixView);

	if (optixView->isCameraMoving) {
		glm::vec2 mouseDelta = glm::vec2(x, y) - optixView->window->lastCursorPosition;

		Camera* camera = optixView->camera.get();
		glm::vec3 rotationDiff = glm::vec3(mouseDelta.y, -mouseDelta.x, 0) * 0.1f;

		camera->rotation += rotationDiff;
		camera->rotation.x = glm::clamp<float>(camera->rotation.x, -80, 80);
	}


	optixView->window->lastCursorPosition = glm::vec2(x, y);
}

static void OnMouseButton(GLFWwindow* window, int button, int action, int mods) {
	OptixView* optixView = static_cast<OptixView*>(glfwGetWindowUserPointer(window));
	assert(optixView);

	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		optixView->isCameraMoving = true;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		optixView->isCameraMoving = false;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

static void OnKeyPressed(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action != GLFW_PRESS) {
		return;
	}

	OptixView* optixView = static_cast<OptixView*>(glfwGetWindowUserPointer(window));
	assert(optixView);

	//Camera Movement
	Camera* camera = optixView->camera.get();
	glm::vec3 movement = glm::vec3(0);
	if (key == 'w' || key == 'W') {
		movement += camera->GetForward();
	}
	else if (key == 's' || key == 'S') {
		movement -= camera->GetForward();
	}

	if (key == 'd' || key == 'D') {
		movement += camera->GetRight();
	}
	else if (key == 'a' || key == 'A') {
		movement -= camera->GetRight();
	}

	if (key == GLFW_KEY_SPACE) {
		movement += camera->worldUp;
	}
	else if (key == GLFW_KEY_LEFT_SHIFT) {
		movement -= camera->worldUp;
	}

	if (glm::length(movement) != 0) {
		movement = glm::normalize(movement);
	}
	camera->position += movement;
}
#pragma endregion


OptixView::OptixView(OptixViewDefinition viewDef, glm::ivec2 viewSize, Model* model, std::unique_ptr<Camera> camera) {
	this->window = std::unique_ptr<OpenGLWindow>(new OpenGLWindow(viewSize));
	this->optixRenderer = std::unique_ptr<OptixRenderer>(new OptixRenderer(viewDef.ptxPath, model));
	this->viewTexture = std::unique_ptr<OptixViewTexture>(new OptixViewTexture());

	this->camera = std::move(camera);

	this->viewSize = viewSize;
	this->Resize(viewSize);

	//Set this view as a custom Pointer for the window
	glfwSetWindowUserPointer(this->window->window, this);
	glfwSetFramebufferSizeCallback(this->window->window, OnWindowResize);

	glfwSetMouseButtonCallback(this->window->window, OnMouseButton);
	glfwSetCursorPosCallback(this->window->window, OnCursorMove);
	glfwSetKeyCallback(this->window->window, OnKeyPressed);

	if (glfwRawMouseMotionSupported()) {
		glfwSetInputMode(this->window->window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
	}
}

#pragma region Public

void OptixView::Run() {

	//Render Loop
	while (!glfwWindowShouldClose(this->window->window)) {
		//camera->rotation = camera->rotation + glm::vec3(0, 0.3, 0);

		this->optixRenderer->SetCamera(this->camera.get());

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