"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xbf16>
    %4 = "chlo.erf_inv"(%2) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    "func.return"(%4) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xE1BF0E3E07C05C3F1B4090BF26C089BF02C0513D56C06A401DC077BF283F6C401DBF4ABE07C02AC0FF3E96408140BA3F52C033C05C40833F113E9AC041C069C00BC0413ECB4098BF46C0D63F7A40A53F7DBF06C02640D8409A404E3F53C035408EBDA9BFF2BE49BFC73F0C40B1BF8140BABFC3C0EE3FCF3F37C094BFEEBF074019404B40254080C0273F19C06ABFF63E88BF8D401840933FE640C8BF1840973EBFC0124093BF0EBED33FB93F833F5BBF394052C0D1BF87BBE1BEF13F8A3FDEBF9F4082BFFD3E3BBFD4C09F40BEBFA03F3FC086C06BBFCE3E0B3FD03F2ABFBD407FC0A54076C08240283F6BC095C0E1C055401F40EEBF843FD1C09EC013C02A40B7C07CBFDBBF51408AC0A9BF253FB43F2740CD3FD8BF8B3E1A40C2BFC13F804069C097401E402C3F61401540913E833F493FD6408BC0B64039BEA9BF9CC0193F15409BBFE7BE594031C02ABFC33E4F40A63EB93C4EC00D3F0BC095BF8740443F243EA03FC3BF2EBF97C0E140FA3F3E3F2FC0B7BF0DC0BCBFB93D464034C0B7BF4B40BDC088C0F0BF893C5C3F1940DBC00C40C040B8BFA0C018C0043FC33F97C022C08DBC0EC0E13F0B3F9440084058BFFFBF2540633F4D3F1FBF4DC0BFC03BBE68C023C013C0753FE9BF2D407DC0B1BFBF3F20BD554041C0B83F5240AEC02EBEEDBFEC3F0540DF3F88BFD5C0534024C078BF04BE933F423E193F2F3F21BF3FC09440BF40A63E313F0840E8BEC5C08B3D4D3EE24083C0E140074056C0BF3E0BC00FC0E33E09C087C0AA3F3BBF38BF1C3E494018C018C0E93D8FBF1D3F1BC0CDC0E93F82BE96BFA64005C145C019C0A7404DC0B1BF194039BFBF40D2BEFD3D7CBF5340A5BF30BF014068BF90BF00409FC0493D3A408FBFC13FD83F0B40903F8D40B03F47BFAC4037404EC04AC0CBC02FBEDDC0933F8440A9BF01C032BFE33FD1BF8A3F394097BFDE3F39C08F40D53C98BFEA3F9C40BCBFBD3FB0C074BF36402EC0253F4E409740BC3FBBC02840EC3F5B3F913DAA403BBF923FA0BFD0BF7CBF1AC0FEBE26BF87C029C0F93FED3F9740B4C0584046C02FBF89BF943F98C0C7BF59405B40EB40AE3E57BF8DC0E53F4FBF53C012BF664095BF583E96404EC05FBFD4BF"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%1) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFF7FFD3DFF7F853FFF7FFF7FFF7FFF7FFF7F393DFF7FFF7FFF7FBFBF2B3FFF7F1DBF35BEFF7FFF7FF33EFF7FFF7FFF7FFF7FFF7FFF7FFF7F013EFF7FFF7FFF7FFF7F2D3EFF7FFF7FFF7FFF7FFF7FFF7FE4BFFF7FFF7FFF7FFF7F6A3FFF7FFF7F7CBDFF7FE5BE61BFFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F2A3FFF7F9BBFE93EFF7FFF7FFF7FFF7FFF7FFF7FFF7F893EFF7FFF7FFF7FFDBDFF7FFF7FFF7F84BFFF7FFF7FFF7F6FBBD3BEFF7FFF7FFF7FFF7FFF7FF13E48BFFF7FFF7FFF7FFF7FFF7FFF7F9DBFBF3E073FFF7F2EBFFF7FFF7FFF7FFF7FFF7F2B3FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FDBBFFF7FFF7FFF7FFF7F273FFF7FFF7FFF7FFF7F7B3EFF7FFF7FFF7FFF7FFF7FFF7FFF7F313FFF7FFF7F833EFF7F613FFF7FFF7FFF7F25BEFF7FFF7F183FFF7FFF7FD9BEFF7FFF7F2EBFB43EFF7F973EA43CFF7F093FFF7FFF7FFF7F573F123EFF7FFF7F34BFFF7FFF7FFF7F4D3FFF7FFF7FFF7FFF7FA43DFF7FFF7FFF7FFF7FFF7FFF7FFF7F733C853FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFD3EFF7FFF7FFF7F7ABCFF7FFF7F073FFF7FFF7F80BFFF7FFF7F8F3F683F1FBFFF7FFF7F27BEFF7FFF7FFF7FB73FFF7FFF7FFF7FFF7FFF7F0EBDFF7FFF7FFF7FFF7FFF7F1BBEFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FC3BFEBBDFF7F2E3E183F353F22BFFF7FFF7FFF7F973E383FFF7FDABEFF7F773D383EFF7FFF7FFF7FFF7FFF7FB03EFF7FFF7FD53EFF7FFF7FFF7F48BF43BF0B3EFF7FFF7FFF7FCF3DFF7F1D3FFF7FFF7FFF7F6ABEFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F45BFFF7FC3BEE13DDBBFFF7FFF7F37BFFF7F98BFFF7FFF7FFF7F323DFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F5DBFFF7FFF7FFF7FFF7FFF7F1CBEFF7FFF7FFF7FFF7FFF7F3ABFFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7FBD3CFF7FFF7FFF7FFF7FFF7FFF7FB4BFFF7FFF7F273FFF7FFF7FFF7FFF7FFF7FFF7F843F813DFF7F48BFFF7FFF7FFF7FDBBFFF7FF2BE29BFFF7FFF7FFF7FFF7FFF7FFF7FFF7FFF7F35BFFF7FFF7FFF7FFF7FFF7FFF7FFF7F9F3E7EBFFF7FFF7F6CBFFF7F0FBFFF7FFF7F423EFF7FFF7F89BFFF7F"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%0) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

