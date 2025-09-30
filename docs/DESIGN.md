@todo
# Design
## Principles
1) User Agency
- The project should provide the user enough agency to f*ck up.
- A key decision when creating the project was the ability to load/unload the model -- the user needs to understand that the model's state is entirely their responsibility and that this project merely provides an interface to the powerful `llama.cpp`. 
- If they choose to do something that causes an error state in a model, then they should have the power to do that. 

2) This is a Framework
- This project should keep a clear separation of Godot and Llama.
- Only access methods need to know anything at all about Godot.

3) Simplicity is Key
- The interface should be kept simple, yet powerful.
- An opinionated API is a core part of this: Methods like `generate_chat` and `generate_text` are pretty clear entry points. They really only flip a switch, but it simplifies the user's decision-making process.

## Roadmap
@TODO -> tho this could probably be better done in github?