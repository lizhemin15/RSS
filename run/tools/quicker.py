import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import sh
import os
app = FastAPI()


class Item(BaseModel):
    key: str = ''
    source: str = 'gitee'


@app.post("/install_rss")
async def post_install_rss(item: Item):
    auth_result = auth(item.key)
    if auth_result:
        install_rss(item.source)
        return "安装成功"
    else:
        return "认证失败"

def install_rss(source):
    try:
        if source == 'gitee':
            res_name = "RSS"
        elif source == 'github':
            res_name = "RSS-github"

        try:
            print("Removing existing", res_name)
            sh.rm("-rf", res_name)
        except Exception as e:
            print("Error occurred while removing", res_name, "Skipping:", e)

        print("Cloning repository from", source)
        sh.git.clone("https://"+source+".com/lizhemin15/"+res_name+".git")

        os.chdir(res_name)

        print("Uninstalling rss package")
        sh.pip.uninstall("rss", _in="y\n")

        print("Building setup.py")
        sh.python("setup.py", "build")

        print("Installing rss package")
        sh.python("setup.py", "install")
        
        os.chdir("..")

    except sh.ErrorReturnCode as e:
        print("An error occurred while running the command:", e.stderr)
    except Exception as e:
        print("An unexpected error occurred:", e)
    

def auth(key):
    if key == 'FlowGPT':
        return True
    else:
        return False


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8848, reload=True)