from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ConanApplication(ConanFile):
    package_type = "application"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    default_options = {"hwloc/*:shared": True}

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("raylib/5.5")
        self.requires("benchmark/1.9.1")
