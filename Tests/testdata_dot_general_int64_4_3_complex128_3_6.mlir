"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi64>, tensor<3x6xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xcomplex<f64>>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi64>) -> tensor<4x3xcomplex<f64>>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xcomplex<f64>>) -> tensor<3x6xcomplex<f64>>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xcomplex<f64>>, tensor<3x6xcomplex<f64>>) -> tensor<4x6xcomplex<f64>>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xcomplex<f64>>, tensor<4x6xcomplex<f64>>) -> ()
    "func.return"(%7) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi64>, tensor<3x6xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2, 2, 0], [1, -7, 2], [1, 0, 0], [-2, -1, -1]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[(0.24200842812993167,-4.2277593423212938), (4.6221314347930251,-2.6368713850512213), (0.9996221307646882,2.3360293265575738), (3.2033880858224624,-1.8427744360435898), (1.6217486592824315,4.4456865248523183), (1.3062865542229514,-0.6288304361593261)], [(4.7048478719379094,2.8329609492421506), (4.0281345880410759,-1.9223183305631795), (-1.0249897761351336,-7.2861179065970347), (2.240343598370254,-0.97304639342562838), (2.7163762408293133,-0.021080138618234212), (1.4432735509764933,-2.1689301399524181)], [(-1.6578011952794189,7.1119750347799195), (-6.0814849977067862,-1.0478559641506684), (0.40710729983898641,-3.3415380671458967), (-7.614392160286874,-0.79179261562921743), (-1.3292719504901869,3.4314185604416991), (-4.6320136988079312,1.0894738566441142)]]> : tensor<3x6xcomplex<f64>>}> : () -> tensor<3x6xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<4x3xi64>, tensor<3x6xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(8.9256788876159554,14.121440583126889), (-1.1879936935038984,1.4291061089760837), (-4.0492238137996441,-19.244294466309217), (-1.9260889749044168,1.7394560852359229), (2.1892551630937636,-8.9335333269411059), (0.27397399350708396,-3.0801994075861838)], [(-36.007529065994277,-9.8345359174565079), (-35.737780676908073,8.7236450005896984), (8.9887651633885959,46.65577853844502), (-27.707801423343064,3.3849650866773739), (-20.051428927503135,11.456084616063356), (-18.060655700228367,16.732628256795827)], [(0.24200842812993167,-4.2277593423212938), (4.6221314347930251,-2.6368713850512213), (0.9996221307646882,2.3360293265575738), (3.2033880858224624,-1.8427744360435898), (1.6217486592824315,4.4456865248523183), (1.3062865542229514,-0.6288304361593261)], [(-3.531063532918354,-1.4894172993794825), (-7.1909124599203391,8.2439170648162907), (-1.3813617852332292,5.9555973206277839), (-1.0327276097283047,5.4503878811420261), (-4.630601608903989,-12.301711471528101), (0.57616703938553471,2.3371171556269563)]]> : tensor<4x6xcomplex<f64>>}> : () -> tensor<4x6xcomplex<f64>>
    "func.return"(%0) : (tensor<4x6xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

