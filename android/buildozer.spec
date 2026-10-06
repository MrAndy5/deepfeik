[app]
title = deepfeik
package.name = deepfeik
package.domain = com.mrAndy5

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
source.include_patterns = assets/*,deepfeik/**/*

version = 1.0.0
requirements = python3,kivy,opencv,numpy,Pillow

p4a.local_recipes = ./recipes

orientation = portrait
fullscreen = 0

android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,INTERNET
android.api = 34
android.minapi = 29
android.ndk = 25b
android.accept_sdk_license = True
android.archs = arm64-v8a

# NDK build flags for native libs
android.add_jars =

# Buildozer log level (0 = error, 1 = info, 2 = debug)
log_level = 1

[buildozer]
warn_on_root = 1
