#include"Renderer/OptixView.h"
#include"ModelLoading/ModelLoader.h"
#include "GlmHelperMethods.h"


int main()
{
    //World Coordinate System: x = out of the mnonitor, y = up, z = left
    const int maxSamples = 100;
    const int maxBounces = 3;//more as "max Collisions per path"

    //std::string modelPath{ "assets/Models/Sponza/" };
    std::string modelPath{ "assets/Models/Test Scene 1/" };
    std::unique_ptr<Model> model = ModelLoader::LoadModel(modelPath, "untitled.gltf");

    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/Renderer/OptiX/devicePrograms.cu.ptx";


    std::unique_ptr<Camera> camera = std::unique_ptr<Camera>(new Camera());

    std::vector<PointLight> pointLights;

    //Sponza Middle
    //camera->SetBlenderPosition(glm::vec3(-0.977644, -0.366231, 1.0745));
    //camera->SetBlenderRotation(glm::vec3(89.1897, 20, 77.765));
    //pointLights.push_back(PointLight{ glm::vec3(0, 2, 0), glm::vec3(500) });

    //Test Scene 1
    camera->SetBlenderPosition(glm::vec3(3.85382f, 0.0f, 1.0f));
    camera->SetBlenderRotation(glm::vec3(90.0f, 0.0f, 90.0f));

    glm::vec3 lightColor = glm::vec3(1);
    //pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), glm::vec3(100)});
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 0.299367f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, 0.7f, 1.69937f), lightColor });
    pointLights.push_back(PointLight{ GlmHelper::BlenderToEnginePosition(1.33906f, -0.7f, 1.69937f), lightColor });

    glm::ivec2 viewSize{ 1920, 1080 };
    //glm::ivec2 viewSize{ 1920 / 2, 1080 / 2 };
    OptixView* optixView = new OptixView(viewDef, viewSize, model.get(), std::move(camera), maxSamples);

    optixView->optixRenderer->SetLights(&pointLights);
    optixView->optixRenderer->SetMaxBounces(maxBounces);

    optixView->Run();
}
