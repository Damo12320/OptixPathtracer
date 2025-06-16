#include"Renderer/OptixView.h"
#include"ModelLoading/ModelLoader.h"
#include "GlmHelperMethods.h"

int main()
{
    const int maxSamples = -1;

    //std::string modelPath{ "assets/Models/3/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Sponza/" };
    std::string modelPath{ "assets/Models/Test Scene 1/" };
    //std::string modelPath{ "assets/Models/Cube + Sphere/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Cube + Sphere Yup/untitled.gltf" };
    //std::string modelPath{ "assets/Models/Cube/untitled.gltf" };
    std::unique_ptr<Model> model = ModelLoader::LoadModel(modelPath, "untitled.gltf");

    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/Renderer/OptiX/devicePrograms.cu.ptx";


    std::unique_ptr<Camera> camera = std::unique_ptr<Camera>(new Camera());
    //camera.position = glm::vec3(-0.977644f, 1.0745f, 0.366231f);
    //camera.rotation = glm::vec3(90 - 89.1897, 180 + 77.765, 0);

    std::vector<PointLight> pointLights;

    //Sponza Middle
    //camera->SetBlenderPosition(glm::vec3(-0.977644, -0.366231, 1.0745));
    //camera->SetBlenderRotation(glm::vec3(89.1897, 20, 77.765));
    //pointLights.push_back(PointLight{ glm::vec3(0, 2, 0), glm::vec3(5000) });

    //Test Scene 1
    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(20);
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 1.69937f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 1.69937f), lightColor });

    glm::ivec2 viewSize{ 1920, 1080 };
    //glm::ivec2 viewSize{ 1920 / 2, 1080 / 2 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model.get(), std::move(camera), maxSamples);

    optixView->optixRenderer->SetLights(&pointLights);

    optixView->Run();
}
