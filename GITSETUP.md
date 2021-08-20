## Guide for Setting up Git
Setting Git on Windows is a bit tedious but I will try to explain it as best as I can.

### Intro
For proects like these, Git is very useful to have. It tracks the project files for changes and updates. It can follow each version of your code and store it's history. Everyone in the project can see what each of us done during the development. We can all work together on the same project and see our codes.

### Terms
Git has some terms. Here I will explain them.

    repository: Basically a project folder.
    clone: Download repository from Github to your local machine.
    add: Track your files and changes in git.
    commit: Save your files to git.
    push: Upload your commit to Github.
    pull: Download the latest project files.

### Goal
We will mostly get the latest project files before working. After doing some work and writing some code we will then commit and push these changes to Github. This way, each of us will see where we are at with the project.

### Installing Git
1. Download the setup file.
[Download Git for Windows](https://git-scm.com/downloads)
2. Click next all the way to the end.
3. After the installiation open cmd.exe and check if Git exists.
    ```bash
    git --version
    ```
    If you get something like 'git version 2.33.0.windows.1' Then it is working.

### Configuring Git for your Github account
Git uses SSH for connecting to your account. We need to create private and public SSH keys and register them. Also Git has it's own config so we need to do that as well.

1. Open 'git bash'
    ```text
    Open Windows search and just type 'git bash' then press enter
    ```

2. Type your name (you can write whatever you want here)
    ```bash
    git config --global user.name "Your Name"
    ```

3. Type your email adress (needs to be your Github mail)
    ```bash
    git config --global user.email "your_email@example.com"
    ```
4. Now we need to setup SSH for authentication and we are done.
   Just press enter if it asks anything. Do not forget to put your email.
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
5. You have a private and a public key now. YAY!
6. Copy your public key with this command.
    ```bash
    clip < ~/.ssh/id_rsa.pub
    ```
7. Now open up Github from the browser.
8. Click to your profile picture and go to Settings.
9. From the left navigation panel find 'SSH and GPG keys'.
10. Click 'New SSH key'
11. Enter some title. (Does not matter but you can type Windows 10 or smth.)
12. Paste the public key you copied earlier. (CTRL + V)
13. Finally click 'Add SSH key' and we are done!

### Using Git from Visual Studio Code for our project
1. From the left panel click on the Source Control. (3rd icon)
2. Click 'Clone Repository' and a textbox will open at the top.
3. Paste the repository URL here.
    ```text
    https://github.com/astral73/Project_LeXuS
    ```
4. Now from the explorer choose a location for the project.
5. After selecting it will download all the files and we have our project.

### Doing some changes and submitting them
Up until this point everything we did was one-time only. We do not need to these step everytime we want to do something for the project.

Before each coding session we should refresh our project files just in case. This way we will have the latest project files.
We can do this refreshing by clicking on the circular icon at the bottom-left. If it asks anything just say yes.

After writing some code and adding files we need to submit it to Github. Submitting has two steps *comitting* and *pushing*.

*Commiting* is basically saying Git that this file is ready for the upload.
*Pushing* is uploading these committed files to Github.

We commit the modified files by:
1. Opening Source Control from the left side (3rd icon)
2. Clicking the tick icon '✓'. If it asks something just say 'Yes' or 'Always yes'.

We are now ready to upload these files to Github:
1. Click the 'three dots' icon near the tick icon '✓'.
2. Then just click Push and voila!

### Final words
These steps might be too much work but it will worth it when our project grows bigger. Also most of these steps are just done once. We do not have to do them again.