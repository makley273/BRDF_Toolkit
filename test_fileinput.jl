"""
Quick test to see what FileInput returns
"""

using Bonito
import Bonito.TailwindDashboard as D

app = App() do session
    file_input = D.FileInput()

    status = Observable("No file uploaded yet")

    on(file_input.value) do file_data
        if isnothing(file_data)
            status[] = "File data is nothing"
        else
            status[] = "Received: $(typeof(file_data))\nKeys: $(keys(file_data))\nFull: $file_data"
        end
    end

    return DOM.div(
        DOM.h1("FileInput Test"),
        file_input,
        DOM.pre(map(x -> DOM.code(x), status))
    )
end

println("Starting test server at http://127.0.0.1:8001")
server = Bonito.Server(app, "127.0.0.1", 8001)
