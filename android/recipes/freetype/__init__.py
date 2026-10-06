from pythonforandroid.recipe import Recipe
from pythonforandroid.util import current_directory
import sh


class FreetypeRecipe(Recipe):
    version = '2.13.2'
    url = 'https://downloads.sourceforge.net/project/freetype/freetype2/2.13.2/freetype-2.13.2.tar.gz'
    built_libraries = {'libfreetype.so': 'objs/.libs'}

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        return env

    def build_arch(self, arch):
        env = self.get_recipe_env(arch)
        with current_directory(self.get_build_dir(arch.arch)):
            sh.bash(
                './configure',
                f'--host={arch.command_prefix}',
                '--prefix=' + self.ctx.get_python_install_dir(arch.arch),
                '--without-zlib',
                '--without-png',
                '--without-bzip2',
                '--enable-shared',
                '--disable-static',
                _env=env,
            )
            sh.make('-j', str(self.ctx.num_cores), _env=env)
            self.install_libs(arch, 'objs/.libs/libfreetype.so')


recipe = FreetypeRecipe()
