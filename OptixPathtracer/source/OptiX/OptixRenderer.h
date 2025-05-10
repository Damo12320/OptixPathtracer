#pragma once

#include "../3rdParty/OptixSample/CUDABuffer.h"
#include "LaunchParams.h"

class OptixRenderer {
public:
	OptixRenderer(const std::string& ptxPath);

    // resize frame buffer to given resolution
    void Resize(glm::ivec2& newSize);

    void Render(uint32_t h_pixels[]);

private:
    /*! helper function that initializes optix and checks for errors */
    void InitOptix();

    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void CreateContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void CreateModule(const std::string& ptxPath);

    /*! does all setup for the raygen program(s) we are going to use */
    void CreateRaygenPrograms();

    /*! does all setup for the miss program(s) we are going to use */
    void CreateMissPrograms();

    /*! does all setup for the hitgroup program(s) we are going to use */
    void CreateHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void CreatePipeline();

    /*! constructs the shader binding table */
    void BuildSBT();

private:
    // device context and stream that optix pipeline will run on, as well as device properties for this device
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;

    // the optix context that our pipeline will run in
    OptixDeviceContext optixContext;

    // the pipeline we're building
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    // the module that contains our device programs
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    //vector of all our program(group)s, and the SBT built around them
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    // our launch parameters, on the host, and the buffer to store them on the device
    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;

    CUDABuffer colorBuffer;
};