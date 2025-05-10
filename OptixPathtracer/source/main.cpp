#include"OptixView.h"

int main()
{
    OptixViewDefinition viewDef;
    viewDef.ptxPath = "source/OptiX/devicePrograms.cu.ptx";

    glm::ivec2 viewSize{ 1920, 1080 };
    OptixView* optixView = new OptixView(viewDef, viewSize);
    optixView->Run();
}
