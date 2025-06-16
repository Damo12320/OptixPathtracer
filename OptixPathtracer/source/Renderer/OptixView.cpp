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

		optixView->clearFrameBuffer = true;
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
		optixView->clearFrameBuffer = true;
	}
	camera->position += movement;
}
#pragma endregion


OptixView::OptixView(OptixViewDefinition viewDef, glm::ivec2 viewSize, Model* model, std::unique_ptr<Camera> camera, int maxSamples) {
	this->maxSamples = maxSamples;

	this->window = std::unique_ptr<OpenGLWindow>(new OpenGLWindow(viewSize));
	this->optixRenderer = std::unique_ptr<OptixRenderer>(new OptixRenderer(viewDef.ptxPath, model));

	//Texture to store the new Frame
	this->newFrame = std::unique_ptr<GLTexture2D>(new GLTexture2D(viewSize));

	//FrameBuffer
	this->framebuffer = std::unique_ptr<Framebuffer>(new Framebuffer());
	this->framebuffer->AttachNewTexture2D(GL_COLOR_ATTACHMENT0, viewSize);

	if (!this->framebuffer->IsComplete()) {
		std::cout << "ERROR::OPTIXVIEW::CONSTRUCTOR::FRAMEBUFFER_NOT_COMPLETE" << std::endl;
	}

	//Shader
	this->combineShader = std::unique_ptr<ShaderProgramm>(new ShaderProgramm("source/Renderer/OpenGL/Shader/PostProcess.vert", "source/Renderer/OpenGL/Shader/AddPathtracedFrame.frag"));
	this->finalShader = std::unique_ptr<ShaderProgramm>(new ShaderProgramm("source/Renderer/OpenGL/Shader/PostProcess.vert", "source/Renderer/OpenGL/Shader/Final.frag"));

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

		if (this->maxSamples < 0 || this->samples <= this->maxSamples) {
			this->optixRenderer->SetCamera(this->camera.get());

			//Optix
			this->DrawOptix(this->newFrame.get());

			if (this->clearFrameBuffer) {
				this->clearFrameBuffer = false;

				this->framebuffer->Bind();
				glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
				glClear(GL_COLOR_BUFFER_BIT);

				glBindFramebuffer(GL_FRAMEBUFFER, 0);

				this->samples = 0;
			}

			//OpenGL
			glDisable(GL_DEPTH_TEST);
			glViewport(0, 0, this->viewSize.x, this->viewSize.y);

			this->AddNewFrameToBuffer(this->newFrame.get(), this->framebuffer.get());

			if (samples % 10 == 0) {
				std::cout << "at Sample: " << samples << std::endl;
			}
		}
		else if (this->samples == this->maxSamples + 1) {
			std::cout << "Render is finished" << std::endl;
			this->samples++;
		}

		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		this->finalShader->Bind();
		this->finalShader->SetTextureLocation("image", 0);
		this->framebuffer->GetAttachedTexture(GL_COLOR_ATTACHMENT0)->BindToUnit(0);
		//this->newFrame->BindToUnit(0);

		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

		//OpenGL
		//this->GLFrameSetup();
		//this->DrawToWindow();
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

void OptixView::DrawOptix(GLTexture2D* texture) {
	//Pixel Storage
	std::vector<uint32_t> pixels;
	pixels.resize(this->viewSize.x * this->viewSize.y);

	//Render Optix
	this->optixRenderer->Render(pixels.data());

	texture->SetData(pixels.data(), this->viewSize);
}

void OptixView::AddNewFrameToBuffer(GLTexture2D* newFrame, Framebuffer* buffer) {
	//Are they the same size?
	if (newFrame->GetWidth() != buffer->GetSize().x || newFrame->GetHeight() != buffer->GetSize().y) {
		std::cout << "ERROR::OPTIXVIEW::AddNewFrameToBuffer  the buffer and the Texture have differnt sizes!" << std::endl;
		std::cout << "Texture Width: " << newFrame->GetWidth() << " + Height: " << newFrame->GetHeight() << std::endl;
		std::cout << "Buffer Width: " << buffer->GetSize().x << " + Height: " << buffer->GetSize().y << std::endl;
		return;
	}

	//Bind shader
	this->combineShader->Bind();

	//Set Textures
	this->combineShader->SetTextureLocation("frameBufferImage", 1);
	buffer->GetAttachedTexture(GL_COLOR_ATTACHMENT0)->BindToUnit(1);

	this->combineShader->SetTextureLocation("imageToAdd", 2);
	newFrame->BindToUnit(2);

	//Set Weight
	this->samples++;
	float frameWeight = 0;

	bool addContinuously = this->maxSamples < 0;

	if (addContinuously) {
		frameWeight = 1.0f / this->samples;
	}
	else {
		frameWeight = 1.0f / this->maxSamples;
	}

	this->combineShader->SetBool("addContinuously", addContinuously);
	this->combineShader->SetFloat("weight", frameWeight);

	//Bind buffer
	buffer->Bind();

	//Draw
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	//Unbind Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

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

void OptixView::DrawToWindow() {

	//Set viweport size
	glViewport(0, 0, this->viewSize.x, this->viewSize.y);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.f, (float)this->viewSize.x, 0.f, (float)this->viewSize.y, -1.f, 1.f);

	this->framebuffer->GetAttachedTexture(GL_COLOR_ATTACHMENT0)->Bind();
	//this->newFrame->Bind();

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