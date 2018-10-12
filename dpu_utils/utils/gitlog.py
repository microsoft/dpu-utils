import subprocess

__all__ = ['git_tag_run']


def git_tag_run(train_run_id: str)-> str:
    """Tag current version of code in git"""
    cur_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])

    # This call may fail if there are no changes, and we're fine with it:
    subprocess.call(['git', 'commit', '-a', '-m', 'Automatic commit of state for run %s' % train_run_id])
    new_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    subprocess.check_call(['git', 'tag', 'runs/%s' % train_run_id])

    # If the hash changed, we created a new commit (otherwise, no changes existed), so back out of that again
    if cur_hash != new_hash:
        subprocess.check_call(['git', 'reset', '--mixed', 'HEAD^'])

    return new_hash.strip().decode("utf-8")
