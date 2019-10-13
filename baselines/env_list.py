
def get_envlist(env_type):
    """ get list of env names wrt the type of env """
    try:
        l = env_list[env_type]
    except:
        print('Env Type {:s} Not Found!'.format(env_type))
    return l

env_list={
    ## Gym
    'atari':[],    # Atari 
    'box2d':[],    # Box2D
    'classic_control':[      # Classic control
        'Acrobot-v1',
        'CartPole-v1',
        'CartPole-v0',
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'Pendulum-v0'
    ] ,
    'mujoco':[],     # MuJoCo
    'robotics':[],   # Robotics

    ## Deepmind Control Suite
    'dm_control':[],      

    ## RLBench
    'rlbench':[
    'BeatTheBuzz',
    'BlockPyramid',
    'ChangeChannel',
    'ChangeClock',
    'CloseBox',
    'CloseDoor',
    'CloseDrawer',
    'CloseFridge',
    'CloseGrill',
    'CloseJar',
    'CloseLaptopLid',
    'CloseMicrowave',
    'EmptyContainer',
    'EmptyDishwasher',
    'GetIceFromFridge',
    'HangFrameOnHanger',
    'HannoiSquare',
    'HitBallWithQueue',
    'Hockey',
    'InsertUsbInComputer',
    'LampOff',
    'LampOn',
    'LightBulbIn',
    'LightBulbOut',
    'MeatOffGrill',
    'MeatOnGrill',
    'MoveHanger',
    'OpenBox',
    'OpenDoor',
    'OpenDrawer',
    'OpenFridge',
    'OpenGrill',
    'OpenJar',
    'OpenMicrowave',
    'OpenOven',
    'OpenWindow',
    'OpenWineBottle',
    'PhoneOnBase',
    'PickAndLift',
    'PickUpCup',
    'PlaceCups',
    'PlaceHangerOnRack',
    'PlaceShapeInShapeSorter',
    'PlayJenga',
    'PlugChargerInPowerSupply',
    'PourFromCupToCup',
    'PressSwitch',
    'PushButton',
    'PushButtons',
    'PutBooksOnBookshelf',
    'PutBottleInFridge',
    'PutGroceriesInCupboard',
    'PutItemInDrawer',
    'PutKnifeInKnifeBlock',
    'PutKnifeOnChoppingBoard',
    'PutMoneyInSafe',
    'PutPlateInColoredDishRack',
    'PutRubbishInBin',
    'PutShoesInBox',
    'PutToiletRollOnStand',
    'PutTrayInOven',
    'PutUmbrellaInUmbrellaStand',
    'ReachAndDrag',
    'ReachTarget',
    'RemoveCups',
    'ScoopWithSpatula',
    'ScrewNail',
    'SetTheTable',
    'SetupCheckers',
    'SlideBlockToTarget',
    'SlideCabinetOpen',
    'SlideCabinetOpenAndPlaceCups',
    'SolvePuzzle',
    'StackBlocks',
    'StackCups',
    'StackWine',
    'StraightenRope',
    'SweepToDustpan',
    'TakeCupOutFromCabinet',
    'TakeFrameOffHanger',
    'TakeItemOutOfDrawer',
    'TakeLidOffSaucepan',
    'TakeMoneyOutSafe',
    'TakeOffWeighingScales',
    'TakePlateOffColoredDishRack',
    'TakeShoesOutOfBox',
    'TakeToiletRollOffStand',
    'TakeTrayOutOfOven',
    'TakeUmbrellaOutOfUmbrellaStand',
    'TakeUsbOutOfComputer',
    'ToiletSeatDown',
    'ToiletSeatUp',
    'TurnOvenOn',
    'TurnTap',
    'TvOff',
    'TvOn',
    'UnplugCharger',
    'WaterPlants',
    'WeighingScales',
    'WipeDesk'
    ]
}

