import React, { useEffect, useRef } from 'react';
import { Live2DModel, MotionPriority } from 'pixi-live2d-display';
import AnimationControl from './AnimationControl';
import { roleToLive2dMapper } from '@/constants/live2d';
import { useBasicLayout } from '@/hooks/useBasicLayout';
import { useStore } from '@/store/storeProvider';
import { observer } from 'mobx-react';

interface Live2dModelProps {
  role: string;
  onModelReady?: () => void;
}

const RANDOM_MOTIONS = ['random1', 'random2', 'random3', 'random4', 'random5'];

const Live2dModel: React.FC<Live2dModelProps> = observer(({ role, onModelReady }) => {
  const { live2dStore } = useStore();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<Live2DModel | null>(null);
  const movement = new AnimationControl();
  const appRef = useRef<PIXI.Application | null>(null);
  const { isMobile } = useBasicLayout();

  const animation = (currentTime: number, model: any) => {
    movement.head_movement(currentTime);
    movement.eyes_movement(currentTime);
    movement.eyes_lid_movement(currentTime);

    const params = [
      ['ParamAngleX', movement.head[0]],
      ['ParamAngleY', movement.head[1]],
      ['ParamAngleZ', movement.head[2]],
      ['ParamBodyAngleX', movement.head[0] / 10],
      ['ParamBodyAngleY', movement.head[1] / 10],
      ['ParamBodyAngleZ', movement.head[2] / 10],
      ['ParamEyeBallX', movement.eyes[0]],
      ['ParamEyeBallY', movement.eyes[1]],
      ['ParamEyeLOpen', movement.eye_lids[0]],
      ['ParamEyeROpen', movement.eye_lids[1]],
    ];

    params.forEach(([param, value]) => {
      model.internalModel.coreModel.setParameterValueById(param, value);
    });

    setTimeout(() => {
      animation(Date.now(), model);
    }, 30);
  };

  const generateLive2d = async (role: string) => {
    if (!canvasRef.current) return;
    
    const roleConfig = roleToLive2dMapper[role];
    if (!roleConfig) return;

    const savedConfig = localStorage.getItem('live2d_config');
    const config = savedConfig ? JSON.parse(savedConfig) : roleConfig;

    const app = new PIXI.Application({
      view: canvasRef.current,
      autoStart: true,
      transparent: true,
      resize: true,
      resizeTo: window,
    });
    appRef.current = app;

    const model = await Live2DModel.from(roleConfig.path, { 
      autoInteract: false 
    });
    modelRef.current = model;
    app.stage.addChild(model);
    live2dStore.setModel(model);
    if (!isMobile) {
      model.scale.set(config.scale1);
      model.y = config.y1;
      model.x = config.x1 + (window.innerWidth - 1620) / 2;
    } else {
      model.scale.set(roleConfig.scale2);
      model.y = roleConfig.y2;
      model.x = roleConfig.x2 + (window.innerWidth - 380) / 2;
    }

    PIXI.live2d.config.motionFadingDuration = 0;
    PIXI.live2d.config.idleMotionFadingDuration = 0;
    PIXI.live2d.config.expressionFadingDuration = 0;
    model.internalModel.breath = null;
    const updateFn = model.internalModel.motionManager.update
    model.internalModel.motionManager.update = () => {
      updateFn.call(model.internalModel.motionManager, model.internalModel.coreModel, Date.now() / 1000)
    }
    animation(new Date().getTime(), model);
    window.onresize = () => {
      if (!isMobile) {
        model.scale.set(config.scale1);
        model.y = config.y1;
        model.x = config.x1 + (window.innerWidth - 1620) / 2;
      } else {
        model.scale.set(roleConfig.scale2);
        model.y = roleConfig.y2;
        model.x = roleConfig.x2 + (window.innerWidth - 380) / 2;
      }
    }
    setTimeout(() => {
      onModelReady?.();
    }, 3000);
  };

  const handleEmotion = (emotion: string) => {
    if(emotion === 'neutral') {
      modelRef.current?.internalModel.motionManager.expressionManager.resetExpression();
      return;
    }
    if (!modelRef.current?.internalModel?.motionManager?.expressionManager) return;
    modelRef.current.internalModel.motionManager.expressionManager.setExpression(emotion);
  };

  const handleMotion = (motion: string) => {
    if (!modelRef.current?.internalModel?.motionManager) return;
    modelRef.current.internalModel.motionManager.startMotion(motion, 0, MotionPriority.FORCE);
  };

  const playRandomMotion = () => {
    if (!modelRef.current?.internalModel?.motionManager) return;
    const randomIndex = Math.floor(Math.random() * RANDOM_MOTIONS.length);
    const randomMotion = RANDOM_MOTIONS[randomIndex];
    handleMotion(randomMotion);
  };

  const updateModelPosition = (config: any) => {
    if (!modelRef.current) return;
    
    if (!isMobile) {
      modelRef.current.scale.set(config.scale1);
      modelRef.current.y = config.y1;
      modelRef.current.x = config.x1 + (window.innerWidth - 1620) / 2;
    } else {
      modelRef.current.scale.set(roleConfig.scale2);
      modelRef.current.y = roleConfig.y2;
      modelRef.current.x = roleConfig.x2 + (window.innerWidth - 380) / 2;
    }
  };

  useEffect(() => {
    if (role) {
      generateLive2d(role);
    }
  }, [role]);

  useEffect(() => {
    if (live2dStore.emotion) {
      handleEmotion('neutral');
      if(!live2dStore.emotion.includes('random')) {
        handleEmotion(live2dStore.emotion);
      }else{
        handleEmotion('neutral');
      }
    }
  }, [live2dStore.emotion]);

  useEffect(() => {
    if (live2dStore.motion) {
      if (live2dStore.motion === 'speaking' || live2dStore.motion === 'thinking') {
        playRandomMotion();
      } else {
        handleMotion(live2dStore.motion);
      }
    }
  }, [live2dStore.motion]);

  useEffect(() => {
    const handleConfigChange = (e: StorageEvent) => {
      if (e.key === 'live2d_config' && e.newValue) {
        const newConfig = JSON.parse(e.newValue);
        updateModelPosition(newConfig);
      }
    };

    window.addEventListener('storage', handleConfigChange);
    return () => window.removeEventListener('storage', handleConfigChange);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left:0,
        right:0,
        bottom:0,
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
      }}
    />
  );
});

export default Live2dModel;
