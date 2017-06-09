import shutil

WARNING = """
Warning: Delete all generated folders!
"""
print(WARNING)
shutil.rmtree('tmp', ignore_errors=True)
