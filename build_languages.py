import os
import subprocess
import shutil
from tree_sitter import Language

def build_tree_sitter_library():
    grammar_path = 'vendor/tree-sitter-python'
    output_path = 'build/my-languages.dll' if os.name == 'nt' else 'build/my-languages.so'
    
    # Ensure build directory exists
    os.makedirs('build', exist_ok=True)
    
    # Verify grammar directory
    if not os.path.exists(grammar_path):
        print(f"Error: {grammar_path} does not exist. Run: git clone https://github.com/tree-sitter/tree-sitter-python {grammar_path}")
        return False
    
    # Run tree-sitter generate
    try:
        subprocess.run(['npx', 'tree-sitter', 'generate'], cwd=grammar_path, check=True, shell=True)
        print(f"Generated grammar in {grammar_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate grammar: {str(e)}")
        print("Ensure tree-sitter-cli is installed (npm install -g tree-sitter-cli) and in PATH.")
        return False
    except FileNotFoundError:
        print("Error: npx or tree-sitter not found. Ensure Node.js and tree-sitter-cli are installed.")
        return False
    
    # Try building with Language.build_library
    try:
        Language.build_library(output_path, [grammar_path])
        print(f"Successfully built {output_path}")
        return True
    except AttributeError:
        print("Language.build_library not available. Attempting manual compilation...")
    
    # Manual compilation with cl.exe (Windows)
    if os.name == 'nt':
        try:
            cl_exe = 'cl'
            # Check if cl.exe is available
            if not shutil.which(cl_exe):
                print("Error: cl.exe not found. Ensure Visual C++ Build Tools are installed and in PATH.")
                return False
            src_path = os.path.join(grammar_path, 'src')
            parser_c = os.path.join(src_path, 'parser.c')
            scanner_c = os.path.join(src_path, 'scanner.c') if os.path.exists(os.path.join(src_path, 'scanner.c')) else None
            cmd = [cl_exe, '/LD', '/I', src_path, parser_c]
            if scanner_c:
                cmd.append(scanner_c)
            cmd.append('/link')
            cmd.append(f'/OUT:{output_path}')
            subprocess.run(cmd, check=True, shell=True)
            print(f"Manually compiled {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Manual compilation failed: {str(e)}")
            print("Ensure Visual C++ Build Tools are installed and cl.exe is in PATH.")
            return False
        except Exception as e:
            print(f"Unexpected error during compilation: {str(e)}")
            return False
    else:
        # Unix: Use gcc
        try:
            gcc = 'gcc'
            src_path = os.path.join(grammar_path, 'src')
            parser_c = os.path.join(src_path, 'parser.c')
            scanner_c = os.path.join(src_path, 'scanner.c') if os.path.exists(os.path.join(src_path, 'scanner.c')) else None
            cmd = [gcc, '-shared', '-o', output_path, '-I', src_path, parser_c]
            if scanner_c:
                cmd.append(scanner_c)
            subprocess.run(cmd, check=True, shell=True)
            print(f"Manually compiled {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Manual compilation failed: {str(e)}")
            return False

if __name__ == "__main__":
    if build_tree_sitter_library():
        print("Tree-sitter setup complete. You can now run the RepoScope agent.")
    else:
        print("Tree-sitter setup failed. The agent will use regex-based parsing as a fallback.")