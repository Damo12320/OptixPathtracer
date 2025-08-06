#include"Renderer/OptixView.h"
#include"ModelLoading/ModelLoader.h"
#include "GlmHelperMethods.h"

//Spheres
void Scene1(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/1/";
    fileName = "untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(1);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 1.69937f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 1.69937f), lightColor });
}

//Sponza Middle
void Scene2(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/2/";
    fileName = "untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(-0.977644f, -0.366231f, 1.0745f));
    camera->SetBlenderRotation(glm::vec3(89.1897f, 0.0f, 77.765f));

    glm::vec3 lightColor = glm::vec3(100);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(0.0f, 0.0f, 4.12939f), lightColor });
}

//Cornell Diffuse
void Scene3(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/3/";
    fileName = "untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(1);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(0, 0, 1.69221f), lightColor });
}

//Dragon Glossy
void Scene4(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/4/";
    fileName = "Untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(1);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(0, 0, 1.69221f), lightColor });
}

//Dragon Layered
void Scene5(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/5/";
    fileName = "Untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(1);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(0, 0, 1.69221f), lightColor });
}

//Sponza Up
void Scene6(std::string& modelPath, std::string& fileName, Camera* camera, std::vector<PointLight>& pointLights) {
    modelPath = "assets/Models/TestScenes/2/";
    fileName = "untitled.gltf";

    camera->SetBlenderPosition(glm::vec3(10.3184f, 3.66455f, 5.19961f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(100);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(0.0f, 0.0f, 4.12939f), lightColor });
}


int main()
{
    //right Handed Coordinate System
    //World Coordinate System: x = out of the mnonitor, y = up, z = left
    //const int maxSamples = 1024;
    const int maxSamples = 1;
    const int maxBounces = 2;//more as "max Collisions per path"


    std::string modelPath;
    std::string fileName;
    std::unique_ptr<Camera> camera = std::unique_ptr<Camera>(new Camera());
    std::vector<PointLight> pointLights;

    //Scene1(modelPath, fileName, camera.get(), pointLights);
    //Scene2(modelPath, fileName, camera.get(), pointLights);
    //Scene3(modelPath, fileName, camera.get(), pointLights);
    //Scene4(modelPath, fileName, camera.get(), pointLights);
    //Scene5(modelPath, fileName, camera.get(), pointLights);
    Scene6(modelPath, fileName, camera.get(), pointLights);

    //std::string modelPath{ "assets/Models/Test Scene 1/" };
    std::unique_ptr<Model> model = ModelLoader::LoadModel(modelPath, fileName);

    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/Renderer/OptiX/devicePrograms.cu.ptx";

    //Sponza Middle
    //camera->SetBlenderPosition(glm::vec3(-0.977644, -0.366231, 1.0745));
    //camera->SetBlenderRotation(glm::vec3(89.1897, 20, 77.765));
    //pointLights.push_back(PointLight{ glm::vec3(0, 2, 0), glm::vec3(500) });

    ////Test Scene 1
    //camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    //camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));
    //
    ////glm::vec3 lightColor = glm::vec3(1);
    //glm::vec3 lightColor = glm::vec3(1);
    ////pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), glm::vec3(100)});
    //pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), lightColor });
    //pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 0.299367f), lightColor });
    //pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 1.69937f), lightColor });
    //pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 1.69937f), lightColor });

    glm::ivec2 viewSize{ 1920, 1080 };
    //glm::ivec2 viewSize{ 1920 / 2, 1080 / 2 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model.get(), std::move(camera), maxSamples);

    optixView->optixRenderer->SetLights(&pointLights);
    optixView->optixRenderer->SetMaxBounces(maxBounces);

    optixView->Run();
}
